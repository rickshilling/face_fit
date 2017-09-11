import eos
import tensorflow as tf

model_file_name = "../../share/sfm_shape_3448.bin"

model = eos.morphablemodel.load_model(model_file_name)
shape_model = model.get_shape_model()
shape_model_pca_basis = tf.Variable(shape_model.get_orthonormal_pca_basis(), name='PCA_basis')

pca_matrix_size = shape_model_pca_basis.get_shape().as_list()
x = tf.Variable(tf.random_normal([pca_matrix_size[1], 1], 0.01), name='coefficients')
