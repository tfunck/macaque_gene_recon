from brainbuilder.reconstruct import reconstruct

hemi_info_csv = 'data/hemisphere.csv'
chunk_info_csv = 'data/chunk.csv'
sect_info_csv = 'data/section.csv'

reconstruct(
        hemi_info_csv,
        chunk_info_csv,
        sect_info_csv,
        resolution_list=[4.6, 2.8, 1.4, 0.7],
        output_dir='outputs/reconstruction/',
        pytorch_model_dir=None,
        n_depths=10,
        use_nnunet=False,
        num_cores=1,
)