import eos
import tensorflow as tf

model_file_name = "../../share/sfm_shape_3448.bin"

model = eos.morphablemodel.load_model(model_file_name)
shape_model = model.get_shape_model()
shape_model_pca_basis = tf.Variable(shape_model.get_orthonormal_pca_basis(), name='PCA_basis')

pca_matrix_size = shape_model_pca_basis.get_shape().as_list()

# Model & estimate
x = tf.Variable(tf.random_normal([pca_matrix_size[1], 1], stddev=1e-2), name='coefficients')
b = tf.matmul(shape_model_pca_basis,x)
x_hat = tf.matrix_solve_ls(shape_model_pca_basis,b)
loss = tf.norm(x-x_hat)
model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    print(session.run(loss))

