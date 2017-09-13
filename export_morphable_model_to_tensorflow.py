# From "Huber, 'A Multiresolution 3D Morphable Face Model and Fitting Framework'"
import eos
import tensorflow as tf

model_file_name = "../../share/sfm_shape_3448.bin"

model = eos.morphablemodel.load_model(model_file_name)
shape_model = model.get_shape_model()
shape_model_pca_basis = tf.Variable(shape_model.get_rescaled_pca_basis(), name='PCA_shape')

pca_shape_matrix_size = shape_model_pca_basis.get_shape().as_list()
    
mean = shape_model.get_mean()
v_bar = tf.Variable(mean.reshape([mean.shape[0], 1]),name='shape_mean')
alpha = tf.Variable(tf.random_normal([pca_shape_matrix_size[1], 1], stddev=1e-2), name='coefficients')
v = tf.matmul(shape_model_pca_basis,alpha) + v_bar
alpha_hat = tf.matrix_solve_ls(shape_model_pca_basis,v-v_bar)
loss = tf.norm(alpha-alpha_hat)
global_model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(global_model)
    print(session.run(loss))
