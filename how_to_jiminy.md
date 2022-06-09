First, there is an example of simple RL environment based on Jiminy [here](https://github.com/duburcqa/jiminy/blob/master/python/gym_jiminy/envs/gym_jiminy/envs/cartpole.py). Next, a basic tutorial is available [here](https://github.com/duburcqa/jiminy/blob/master/examples/python/tutorial.ipynb). Finally, a sketch of document can be found [here](https://duburcqa.github.io/jiminy/).

Regarding your specific setup, you can add deformation points either at the location of an actual joint or a fixed frame. In you case, it seems to be fixed frame. You need to adapt your kinematic tree to break up the single link in several smaller fixed rigidly attached to each other using "fixed" joints. Each of the sub-links must have their own visual and collision geometries. Then, once you URDF model is adapted you can start simulating flexibility very easily. To this end, two options must be configured:

    model_options = robot.get_model_options()
    model_options["dynamics"]["enableFlexibleModel"] = True
    model_options["dynamics"]["flexibilityConfig"] = [{
        'frameName': "PendulumJoint",
        'stiffness': k * np.ones(3),
        'damping': nu * np.ones(3),
        'inertia': np.zeros(3)
    }]
    robot.set_model_options(model_options)

where `model_options["dynamics"]["flexibilityConfig"]` is a list of sub-dict that fully characterize a given deformation point. [Here](https://objects.githubusercontent.com/github-production-repository-file-5c1aeb/212770284/8837958?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20220609%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220609T065443Z&X-Amz-Expires=300&X-Amz-Signature=b5f842816cb9ed83581a10bf2f9cd58ae8b1113d02729f0c6458773f150d0bf6&X-Amz-SignedHeaders=host&actor_id=59015432&key_id=0&repo_id=212770284&response-content-disposition=attachment%3Bfilename%3Dflexibility_model.pdf&response-content-type=application%2Fpdf) is a figure that describe the dynamical model. I can give you more details if you like.

Here is a very basic example to create a flexible model:

    import os
    import numpy as np
    from pkg_resources import resource_filename

    import jiminy_py.core as jiminy

    # Get URDF path
    data_dir = resource_filename("gym_jiminy.envs", "data/toys_models/cartpole")
    urdf_path = os.path.join(data_dir, "cartpole.urdf")

    # Instantiate robot
    robot = jiminy.Robot()
    robot.initialize(urdf_path, has_freeflyer=False, mesh_package_dirs=[data_dir])

    # Add motors and sensors
    motor_joint_name = "slider_to_cart"
    encoder_sensors_descr = {
        "slider": "slider_to_cart",
        "pole": "cart_to_pole"
    }
    motor = jiminy.SimpleMotor(motor_joint_name)
    robot.attach_motor(motor)
    motor.initialize(motor_joint_name)
    for sensor_name, joint_name in encoder_sensors_descr.items():
        encoder = jiminy.EncoderSensor(sensor_name)
        robot.attach_sensor(encoder)
        encoder.initialize(joint_name)

    # Add deformation points
    k, nu, Ia = 20.0, 0.1, 0.0
    model_options = robot.get_model_options()
    model_options["dynamics"]["enableFlexibleModel"] = True
    model_options["dynamics"]["flexibilityConfig"] = [{
        'frameName': "PendulumJoint",
        'stiffness': k * np.ones(3),
        'damping': nu * np.ones(3),
        'inertia': Ia * np.ones(3)
    }]
    robot.set_model_options(model_options)

I recommend you to have a look to the set of all available options for both the engine and the model using get_options. I would give an overview of a subset of the features offered by Jiminy.