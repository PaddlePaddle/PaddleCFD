# import torch
#
# if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
#     arch_list = torch.cuda.get_arch_list()
#     filtered_arch_list = [arch.replace('sm_', '') for arch in arch_list if not arch.startswith('compute_')]
# else:
#     filtered_arch_list = ['75', '86', '90']
#
# unique_arch_set = set(filtered_arch_list)
#
# sorted_unique_arch_list = sorted(list(unique_arch_set))
#
# result_string = ';'.join(sorted_unique_arch_list)
#
# print(result_string)


def run():
    return "70;75;86;90"


if __name__ == "__main__":
    print(run())
