from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from clearml import Task
task = Task.init(project_name='MNIST', task_name='mnist_task')

optimizer = HyperParameterOptimizer(
    base_task_id='92397e8237ae4653a0f3463c26035a06',
    # setting the hyper-parameters to optimize
    hyper_parameters=[
        UniformIntegerParameterRange('number_of_epochs', min_value=2, max_value=12, step_size=2),
        UniformIntegerParameterRange('batch_size', min_value=2, max_value=16, step_size=2),
        UniformParameterRange('dropout1', min_value=0, max_value=0.5, step_size=0.05),
        UniformParameterRange('dropout2', min_value=0, max_value=0.5, step_size=0.05),
    ],
    # setting the objective metric we want to maximize/minimize
    objective_metric_title='Loss',
    objective_metric_series='Loss',
    objective_metric_sign='min',

    # setting optimizer
    optimizer_class=OptimizerOptuna,

    # Configuring optimization parameters
    max_number_of_concurrent_tasks=2,
    optimization_time_limit=60.,
    compute_time_limit=120,
    total_max_jobs=20,
    min_iteration_per_job=100,
    max_iteration_per_job=1000,
)

optimizer.set_report_period(1) # setting the time gap between two consecutive reports
optimizer.start()
optimizer.wait() # wait until process is done
optimizer.stop()