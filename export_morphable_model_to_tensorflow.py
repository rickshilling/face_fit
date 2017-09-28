# From "Huber, 'A Multiresolution 3D Morphable Face Model and Fitting Framework'"
import eos
import tensorflow as tf

model_file_name = "../../share/sfm_shape_3448.bin"

def read_pts(filename):
    """A helper function to read ibug .pts landmarks from a file."""
    lines = open(filename).read().splitlines()
    lines = lines[3:71]

    landmarks = []
    for l in lines:
        coords = l.split()
        landmarks.append([float(coords[0]), float(coords[1])])

    return landmarks

morphable_model = eos.morphablemodel.load_model(model_file_name)
shape_model = morphable_model.get_shape_model()
shape_model_pca_basis = tf.Variable(shape_model.get_rescaled_pca_basis(), name='PCA_shape1')

pca_shape_matrix_size = shape_model_pca_basis.get_shape().as_list()
num_points = pca_shape_matrix_size[0]/3
    
mean = shape_model.get_mean()
v_bar = tf.Variable(mean.reshape([mean.shape[0], 1]),name='shape_mean')
alpha = tf.Variable(tf.random_normal([pca_shape_matrix_size[1], 1], stddev=1e-2), name='coefficients')
v = tf.matmul(shape_model_pca_basis,alpha) + v_bar
alpha_hat = tf.matrix_solve_ls(shape_model_pca_basis,v-v_bar)
loss = tf.norm(alpha-alpha_hat)

# Based on notation in https://en.wikipedia.org/wiki/3D_projection
ds = tf.reshape(v,[3,num_points])
ds_1 = tf.concat([ds, tf.ones([1,num_points])], 0)
focal_length = 2 # TODO: Make tf.Variable
projection_matrix = tf.Variable([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,1./focal_length,0.]])
f = tf.matmul(projection_matrix,ds_1)
b_0 = tf.divide(f[0],f[3])
b_1 = tf.divide(f[1],f[3])
b = tf.concat([b_0, b_1], 0)

global_model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(global_model)
    print(session.run(loss))
    print(session.run(f))
    print(session.run(b))

tf.reset_default_graph()
session.close()
             
