import eos

model_file_name = "../../share/sfm_shape_3448.bin"

model = eos.morphablemodel.load_model(model_file_name)
shape_model = model.get_shape_model()