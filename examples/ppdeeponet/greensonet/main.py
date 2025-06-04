import paddle
from problem import Stokes
from options import Options
from trainer import Trainer


if __name__ == '__main__':
    # python -u main_stokes.py 2>&1 | tee out.log
    args = Options().parse()
    args.pde_case = 'Stokes'
    args.mesh_path = './mesh/stokes/domain.mphtxt'
    args.boundary_mesh_path = ('./mesh/stokes/boundary.mphtxt')
    args.velocity_component_name = 'x'
    args.problem = Stokes('./data/stokes/train_data', args.velocity_component_name)

    args.resume = False
    args.save_vtk = False
    
    args.ngs_boundary = 3
    args.ngs_interior = 4
    args.shape = [1, 3]
    args.domain = [0, 1, 0, 1, 0, 1]
    args.blocks_num = [2, 2, 2]
    args.layers = [[[6, 12, 24, 12, 1], [6, 12, 24, 12, 1], [6, 12, 24, 12, 1]]]

    args.epochs_Adam = 2000
    args.train_samples = 15
    args.test_samples = 5
    args.lr = 0.001

    paddle.seed(seed=args.seed)

    trainer = Trainer(args)
    trainer.train()
