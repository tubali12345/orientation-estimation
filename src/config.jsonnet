{
    pretrained_model_path: "/home/turib/thesis/3_1_dev_split0_multiaccdoa_foa_model.h5",
    trainer_params: {
        max_epochs: 300,
        device: "cuda:2",
        root_dir: "./model_weights_md/",
    },
    lr_scheduler_params: {
        scheduler: "StepLR",
        initial_lr: 0.001,
        frequency: "epoch",
        params: {
            step_size: 6,
            gamma: 0.92,
        },
    },
    data_params: {
        data_dir_path_train: "/ssd2/en_commonvoice_17.0_rs_rm_md",
        data_dir_path_valid: "/ssd2/en_commonvoice_17.0_rs_rm_md",
        dataset_params: {
            duration: 10,
            sr: 44100,
        },
        dataloader_params: {
            batch_size: 16,
            num_workers: 12,
            prefetch_factor: 4,
            pin_memory: false,
        },
        feature_params: {
            fs:44100,
            hop_len_s:0.02,
            label_hop_len_s:0.1,
            max_audio_len_s:60,
            nb_mel_bins:64,
    },
    },
    model_params: {
        modality:'audio', 
        multi_accdoa: false,  
        nb_classes: 2, 
        nb_channels: 6,
        nb_mel_bins:64,

        label_sequence_length:50,    
        batch_size:128,              
        dropout_rate:0.05,          
        nb_cnn2d_filt:64,          
        f_pool_size:[4, 4, 2],     

        nb_heads:8,
        nb_self_attn_layers:2,
        nb_transformer_layers:2,

        nb_rnn_layers:2,
        rnn_size:128,

        nb_fnn_layers:1,
        fnn_size:128, 
    },
}