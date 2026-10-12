from functools import partial

from kappadata.datasets.kd_dataset import KDDataset


class ModeWrapper(KDDataset):
    def __init__(self, dataset: KDDataset, mode: str, return_ctx: bool = False):
        super().__init__()
        self.dataset = dataset
        self.mode = mode
        self.return_ctx = return_ctx
        self.propagate_ctx = return_ctx or dataset.requires_propagate_ctx

        self._getitem_fns = []
        self.items = mode.split(" ")

        # check duplicates in fused_operations
        fused_operations = list(self.dataset.fused_operations)
        flat_fused_operations = [op for fused_ops in fused_operations for op in fused_ops]
        assert all(len(set(fused_ops)) == len(fused_ops) for fused_ops in fused_operations)
        assert len(set(flat_fused_operations)) == len(flat_fused_operations)

        # fuse ops
        self.fused_items = []
        self.fused_to_idxs = []
        if len(self.dataset.fused_operations) > 0:
            temp_items = list(self.items)
            for i, item in enumerate(temp_items):
                if item is None:
                    continue
                for fused_ops in self.dataset.fused_operations:
                    if fused_ops[0] == item:
                        # check if other ops are also included
                        if all(op in temp_items for op in fused_ops[1:]):
                            idxs = []
                            for op in fused_ops:
                                idx = temp_items.index(op)
                                temp_items[idx] = None
                                idxs.append(idx)
                            self.fused_to_idxs.append(idxs)
                            self.fused_items.append("".join(fused_ops))
                            break
                else:
                    self.fused_to_idxs.append(i)
                    self.fused_items.append(item)

        # compose getitem functions
        items = self.fused_items if len(self.fused_items) > 0 else self.items
        for item in items:
            if item == "index":
                self._getitem_fns.append(self._getitem_index)
            elif item.startswith("ctx."):
                self.propagate_ctx = True
                ctx_key = item[len("ctx."):]
                self._getitem_fns.append(partial(self._getitem_from_ctx, ctx_key=ctx_key))
            else:
                fn_name = f"getitem_{item}"
                if len(self.fused_items) > 0:
                    # outer wrappers are currently required to implement the fused getitem to avoid being skipped
                    assert hasattr(type(self.dataset), fn_name), f"{type(self.dataset)} has no method getitem_{item}"
                else:
                    # check that dataset implements non-fused getitem (wrappers can use the getitem of their child)
                    assert hasattr(self.dataset, fn_name), f"{type(self.dataset)} has no method getitem_{item}"
                self._getitem_fns.append(getattr(self.dataset, fn_name))

    @staticmethod
    def has_item(mode, item):
        return item in mode.split(" ")

    @staticmethod
    def add_item(mode, item):
        if ModeWrapper.has_item(mode=mode, item=item):
            return mode
        return f"{mode} {item}"

    @staticmethod
    def get_item_index(mode, item):
        return mode.split(" ").index(item)

    @staticmethod
    def get_item(mode, item, batch):
        if not isinstance(batch, (list, tuple)):
            assert len(mode.split(" ")) == 1
            return batch
        idx = ModeWrapper.get_item_index(mode=mode, item=item)
        return batch[idx]

    @staticmethod
    def set_item(mode, item, batch, value):
        idx = mode.split(" ").index(item)
        return tuple(it if i != idx else value for i, it in enumerate(batch))

    @staticmethod
    def _getitem_index(idx, _=None):
        return idx

    @staticmethod
    def _getitem_from_ctx(_, ctx, ctx_key):
        return ctx[ctx_key]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(len(self))[idx]]
        if isinstance(idx, list):
            return [self[i] for i in idx]
        if idx < 0:
            idx = len(self) + idx

        items = []
        ctx = {} if self.propagate_ctx else None
        for getitem_fn in self._getitem_fns:
            item = getitem_fn(idx, ctx)
            items.append(item)

        # unpack fused items into original order
        if len(self.fused_to_idxs) > 0:
            unpacked_items = [None for _ in range(len(self.items))]
            for i, fused_idxs in enumerate(self.fused_to_idxs):
                if isinstance(fused_idxs, list):
                    # fused item
                    for j, fused_idx in enumerate(fused_idxs):
                        unpacked_items[fused_idx] = items[i][j]
                else:
                    # unfused item
                    unpacked_items[fused_idxs] = items[i]
            items = unpacked_items

        if len(items) == 1:
            # single item -> no tuple
            items = items[0]
        else:
            # multiple items -> wrap into tuple
            items = tuple(items)
        if self.return_ctx:
            return items, ctx
        return items

    def __len__(self):
        return len(self.dataset)

    @property
    def collators(self):
        assert self._collators is None, "register collators on root datset"
        return self.dataset.collators

    @property
    def root_dataset(self):
        # root_dataset is implemented in base class -> not handled in __getattr__
        return self.dataset.root_dataset

    @property
    def fused_operations(self):
        raise RuntimeError("fused_operations should not be called from ModeWrapper")

    @property
    def requires_propagate_ctx(self):
        raise RuntimeError("requires_propagate_ctx should not be called from ModeWrapper")

    def has_wrapper(self, wrapper):
        if self == wrapper:
            return True
        return self.dataset.has_wrapper(wrapper)

    def has_wrapper_type(self, wrapper_type):
        if type(self) == wrapper_type:
            return True
        return self.dataset.has_wrapper_type(wrapper_type)

    @property
    def all_wrappers(self):
        return [self] + self.dataset.all_wrappers

    @property
    def all_wrapper_types(self):
        return [type(self)] + self.dataset.all_wrapper_types

    def get_wrappers_of_type(self, wrapper_type):
        wrappers = self.dataset.get_wrappers_of_type(wrapper_type)
        if type(self) == wrapper_type:
            return [self] + wrappers
        return wrappers

    def worker_init_fn(self, rank, **kwargs):
        self.dataset.worker_init_fn(rank, **kwargs)

    def __getattr__(self, item):
        if item == "dataset":
            return getattr(super(), item)
        if item == "__getitems__":
            # TODO implement getitems
            # newer dataset base implementations can circumvent wrappers here
            # -> disable batched getitems and call getitem instead
            # this occoured when doing DataLoader(dataset) where dataset is ModeWrapper(Subset(...))
            # Subset implements __getitems__ which leads to the fetcher from the DataLoader believing also the
            # ModeWrapper has a __getitems__ and therefore calls it instead of the __getitem__ function
            return None
        return getattr(self.dataset, item)

    def __iter__(self):
        """The dataset base class does not define __iter__, so iterating directly would run endlessly."""
        for i in range(len(self)):
            yield self[i]
