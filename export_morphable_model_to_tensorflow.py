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

with tf.variable_scope('PCA_estimation_from_3D') as scope:
    alpha_hat = tf.matrix_solve_ls(shape_model_pca_basis,v-v_bar)
    loss = tf.norm(alpha-alpha_hat)

# Based on notation in https://en.wikipedia.org/wiki/3D_projection
with tf.variable_scope('3D_to_2D') as scope:
    ds = tf.reshape(v,[3,num_points])
    ds_1 = tf.concat([ds, tf.ones([1,num_points])], 0)
    focal_length = 1 # TODO: Make tf.Variable
    projection_matrix = tf.Variable([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,1./focal_length,0.]])
    f = tf.matmul(projection_matrix,ds_1)
    b_0 = tf.div(f[0],f[3])
    b_1 = tf.div(f[1],f[3])
    b = tf.stack([b_0, b_1])

with tf.variable_scope('Estimate_Projection_matrix') as scope:
    a_placeholder = tf.placeholder(tf.float32, shape = [3, 1], name="a_placeholder")
    b_placeholder = tf.placeholder(tf.float32, shape = [2, 1], name="b_placeholder")
    homogeneous_a_model = tf.concat([a_placeholder, tf.ones([1,1])], 0)
    projection_matrix_model = tf.Variable([[1,0,2,0],[0,1,3,0],[0,0,1,0],[0,0,5,0]],dtype=tf.float32)
    f_model = tf.matmul(projection_matrix_model,homogeneous_a_model)
    b_0_model= tf.divide(f_model[0],f_model[3])
    b_1_model = tf.divide(f_model[1],f_model[3])
    b_model = tf.concat([b_0_model, b_1_model], 0)
    error = tf.norm(b_model-b_placeholder)
    train_operation = tf.train.GradientDescentOptimizer(1e-3).minimize(loss=error)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(init)
    print(sess.run(loss))
    ds_eval = ds.eval()# convert from tf to np
    b_eval = b.eval()
    # See scope 'Estimate_Projection_matrix'
    for i in range(num_points):
        a_value = ds_eval[:,i].reshape((3,1))
        b_value = b_eval[:,i].reshape((2,1)) 
        sess.run([train_operation],feed_dict={a_placeholder:a_value,b_placeholder:b_value})
    [projection_matrix_estimate] = sess.run([projection_matrix_model])
    print projection_matrix_estimate
        
tf.reset_default_graph()
sess.close()
writer.close()
             
