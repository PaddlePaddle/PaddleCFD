import paddle
import paddle.nn as nn


class Integral_Cd(paddle.nn.Layer):
    def __init__(self, layers=None, dropout=False, normalize=False):
        super().__init__()
        if layers == None:
            layers = [2, 32, 32, 1]
        self.n_layers = len(layers) - 1
        self.layers = nn.LayerList()
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(in_features=layers[i],
                                         out_features=layers[i + 1]))
            if i < self.n_layers - 1:                
                if normalize:
                    self.layers.append(paddle.nn.BatchNorm(num_features=layers[i+1]))
                if dropout:
                    self.layers.append(nn.dropout(p=0.2))
                self.layers.append(nn.GELU())         

    
    def forward(self, cd_dict, out_keys=None, ):      

        cd_pred = paddle.to_tensor([cd_dict[f'Cd_{out_keys[0]}_pred'],
                                    cd_dict[f'Cd_{out_keys[1]}_pred']]).cuda(blocking=True)
        for _, layer in enumerate(self.layers):
            cd_pred = layer(cd_pred)
        cd_pred = paddle.Tensor.sigmoid(cd_pred) * (0.6 - 0.1) + 0.1
        cd_dict.update({'Cd_pred_modify': cd_pred})
        return cd_dict
    
    
    def forward_v1(self, pred, truth: paddle.Tensor,                  
                out_channels, data_dict, 
                decode_fn=None, out_keys=None, subsample_train=None):
        pred = pred.transpose(perm=[1, 0])
        truth = truth.transpose(perm=[1, 0])
        cd_dict = self.get_cd(pred, truth,  
                                out_channels, data_dict, 
                                decode_fn=decode_fn, out_keys=out_keys, 
                                subsample_train=subsample_train)        

        cd_pred = paddle.to_tensor([cd_dict[f'Cd_{out_keys[0]}_pred'],
                                    cd_dict[f'Cd_{out_keys[1]}_pred']]).cuda(blocking=True)
        for _, layer in enumerate(self.layers):
            cd_pred = layer(cd_pred)
        cd_dict.update({'Cd_pred_modify': cd_pred})
        return cd_dict
    

    def get_cd(self, pred, truth, out_channels, data_dict, 
               decode_fn, out_keys, subsample_train):
        cd_dict = {'Cd_pred': paddle.to_tensor(data=0.0).cuda(blocking=True), 
                   'Cd_truth': paddle.to_tensor(data=0.0).cuda(blocking=True)}
        truth = []
        for i in range(len(out_keys)):
            key = out_keys[i]
            truth_key = data_dict[key][0][::subsample_train, ...]
            if len(tuple(truth_key.shape)) == 1:
                truth_key = truth_key.reshape((-1, 1))
            truth_key = truth_key[:, :out_channels[i]].transpose(perm=[1, 0])
            truth.append(truth_key)
            st, end = sum(out_channels[:i]), sum(out_channels[:i]) + out_channels[i]
            pred_key = pred[st:end, :]
            if decode_fn is not None:
                pred_decode = decode_fn(pred_key, i)
                truth_decode = decode_fn(truth_key, i)
                if key == 'pressure':
                    drag_weight = data_dict['dragWeight'][0].cuda(blocking=True)
                    drag_weight = drag_weight[::subsample_train]
                elif key == 'wallshearstress':
                    drag_weight = data_dict['dragWeightWss'][0][:out_channels[i], :].cuda(blocking=True)
                    drag_weight = drag_weight[..., ::subsample_train]
                drag_pred = paddle.sum(x=drag_weight * pred_decode)
                drag_truth = paddle.sum(x=drag_weight * truth_decode)
                cd_dict.update({f'Cd_{key}_pred': drag_pred,
                                f'Cd_{key}_truth': drag_truth})
                cd_dict['Cd_pred'] += drag_pred
                cd_dict['Cd_truth'] += drag_truth
        return cd_dict





