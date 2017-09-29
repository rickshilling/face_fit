# From "Huber, 'A Multiresolution 3D Morphable Face Model and Fitting Framework'"
import eos
import tensorflow as tf
import numpy as np

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
mean = shape_model.get_mean()

with tf.variable_scope('morphable_model') as scope:
    shape_model_pca_basis = tf.Variable(shape_model.get_rescaled_pca_basis(), name='PCA_shape1')
    pca_shape_matrix_size = shape_model_pca_basis.get_shape().as_list()
    num_points = pca_shape_matrix_size[0]/3
    v_bar = tf.Variable(mean.reshape([mean.shape[0], 1]),name='3D_mesh_mean')
  
with tf.variable_scope('morphable_instance') as scope:
    alpha = tf.Variable(tf.random_normal([pca_shape_matrix_size[1], 1], stddev=1e-2), name='PCA_coefficients')
    v = tf.matmul(shape_model_pca_basis,alpha) + v_bar

with tf.variable_scope('estimation') as scope:
    alpha_hat = tf.matrix_solve_ls(shape_model_pca_basis,v-v_bar)
    loss = tf.norm(alpha-alpha_hat)

# Based on notation in https://en.wikipedia.org/wiki/3D_projection
with tf.variable_scope('3D_to_2D') as scope:
    ds = tf.reshape(v,[3,num_points])
    ds_1 = tf.concat([ds, tf.ones([1,num_points])], 0)
    focal_length = 1 # TODO: Make tf.Variable
    projection_matrix = tf.Variable([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,1./focal_length,0.]])
    f = tf.matmul(projection_matrix,ds_1)
    b_0 = tf.divide(f[0],f[3])
    b_1 = tf.divide(f[1],f[3])
    b = tf.concat([b_0, b_1], 0)

global_model = tf.global_variables_initializer()
with tf.Session() as session:
    writer = tf.summary.FileWriter('./graphs', session.graph)
    session.run(global_model)
    print(session.run(loss))
    print(session.run(b))

tf.reset_default_graph()
session.close()
writer.close()
             
