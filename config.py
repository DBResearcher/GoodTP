import os


class Config:
    
    debug = True #False
    dumpfile_uniqueid = ''
    seed = 2000
    # seed = 42
    gpu = False
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    root_dir = os.path.dirname(os.path.abspath(__file__)) # ./
    data_dir = os.path.join(root_dir, 'data') # ./data
    snapshot_dir = os.path.join(root_dir, 'exp', 'snapshot')
    
    dataset = ''
    dataset_prefix = ''
    dataset_file = ''
    dataset_trajsimi_dict = ''
    
    min_lon = 0.0
    min_lat = 0.0
    max_lon = 0.0
    max_lat = 0.0
    cell_size = 100
    
    # dataset preprocessing
    min_traj_len = 20
    max_traj_len = 2147483647
    trajsimi_max_traj_len = 200
    trajsimi_min_traj_len = 20
    
    
    # trajsimi effectivness experiments
    trajsimi_measure = ''
    gpu_id = 0
    trajsimi_edr_lcss_eps = 200
    trajsimi_edr_lcss_delta = 60
    trajsimi_timereport_exp = False
    trajsimi_sar_distance_eps = 200 # same to trajsimi_edr_lcss_eps 
    trajsimi_sar_time_eps = 10 
    trajsimi_sar_target_length = 10
    
    

    
    # learned methods
    cell_embedding_dim = 128
    traj_embedding_dim = 128
    
    # TrjSR
    trjsr_imgsize_x_lr = 162
    trjsr_imgsize_y_lr = 128
    trjsr_pixelrange_lr = 2
    
    # TMN
    tmn_pooling_size = 10
    tmn_sampling_num = 20
    
    # TrajGAT
    trajgat_num_head = 8
    trajgat_num_encoder_layers = 3
    trajgat_d_lap_pos = 8
    trajgat_encoder_dropout = 0.01
    trajgat_dataloader_num_workers = 8
    trajgat_qtree_node_capacity = 50
    # trajgat_qtree_node_capacity = 10
    
    # RSTS
    rsts_num_layers = 3
    rsts_bidirectional = True
    rsts_dropout = 0.2

    # Agent
    grid_size = 20
    lambda_penalty = 0.01
    num_iterations = 100
    beta = (0.3, 0.25, 0.25, 0.2)
    alpha = 0.0
    cluster_num = 10
    trajectory_num = 7000
    budget = 1000000
    max_leaf_size = 256
    # node_random_sampling = False
    ablation_type = 0
    training_trajectory_ratio = 1.0

    
    @classmethod
    def post_value_updates(cls):

        if 'porto' == cls.dataset:
            cls.dataset_prefix = 'porto_inf'
            cls.min_lon = -8.7005
            cls.min_lat = 41.1001
            cls.max_lon = -8.5192
            cls.max_lat = 41.2086
            # -15.630759, -3.930948, 36.886104, 45.657225 
            # all raw trajectories in porto 
            
        elif 'chengdu' == cls.dataset:
            cls.dataset_prefix = 'chengdu_inf'
            cls.min_lon = 104.03959949
            cls.min_lat = 30.65529979
            cls.max_lon = 104.12715401
            cls.max_lat = 30.73027283
            
        else:
            pass
        
        cls.dataset_file = os.path.join(cls.data_dir, cls.dataset_prefix)
        
        cls.dataset_trajsimi_traj = '{}_trajsimi_dict_traj_all'.format( \
                                    cls.dataset_file)

    @classmethod
    def update(cls, dic: dict):
        for k, v in dic.items():
            if k in cls.__dict__:
                assert type(getattr(Config, k)) == type(v)
            setattr(Config, k, v)
        cls.post_value_updates()


    @classmethod
    def to_str(cls): # __str__, self
        dic = cls.__dict__.copy()
        lst = list(filter( \
                        lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                        dic.items() \
                        ))
        return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])

