from clearml import Task
from clearml.automation.optuna.optuna import OptimizerOptuna
from clearml.automation.optimization import HyperParameterOptimizer
from clearml.automation.parameters import UniformIntegerParameterRange, DiscreteParameterRange, UniformParameterRange

search_strategy = OptimizerOptuna

def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))

# Connecting CLEARML
task = Task.init(project_name='BBcluster Hyper-Parameter Optimization',
                 task_name='Automatic Hyper-Parameter Optimization',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

# experiment template to optimize in the hyper-parameter optimization
args = {
    'template_task_id': None,
    'run_as_service': False,
}
args = task.connect(args)

# Get the template task experiment that we want to optimize
if not args['template_task_id']:
    args['template_task_id'] = Task.get_task(
        project_name='BB Clustering', task_name='bbclustering_fixed_lambda').id

an_optimizer = HyperParameterOptimizer(
    # This is the experiment we want to optimize
    base_task_id=args['template_task_id'],
    hyper_parameters=[
    UniformIntegerParameterRange('lambda_val', min_value=20, max_value=200, step_size=10),
    UniformParameterRange('reg', min_value=0.0, max_value=10, step_size=0.5)
    ],
    objective_metric_title='val_ARI',
    objective_metric_series='val_ARI',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=2,
    optimizer_class=search_strategy,
    execution_queue='1xGPU',
    # Optional: Limit the execution time of a single experiment, in minutes.
    # (this is optional, and if using  OptimizerBOHB, it is ignored)
    time_limit_per_job=10.,
    # Check the experiments every 6 seconds is way too often, we should probably set it to 5 min,
    # assuming a single experiment is usually hours...
    pool_period_min=0.1,
    # set the maximum number of jobs to launch for the optimization, default (None) unlimited
    # If OptimizerBOHB is used, it defined the maximum budget in terms of full jobs
    # basically the cumulative number of iterations will not exceed total_max_jobs * max_iteration_per_job
    total_max_jobs=10,
    # This is only applicable for OptimizerBOHB and ignore by the rest
    # set the minimum number of iterations for an experiment, before early stopping
    min_iteration_per_job=10,
    # Set the maximum number of iterations for an experiment to execute
    # (This is optional, unless using OptimizerBOHB where this is a must)
    max_iteration_per_job=30
)

# report every 12 seconds, this is way too often, but we are testing here J
an_optimizer.set_report_period(0.2)
# start the optimization process, callback function to be called every time an experiment is completed
# this function returns immediately
an_optimizer.start(job_complete_callback=job_complete_callback)
# set the time limit for the optimization process (2 hours)

# set the time limit for the optimization process (2 hours)
an_optimizer.set_time_limit(in_minutes=90.0)
# wait until process is done (notice we are controlling the optimization process in the background)
an_optimizer.wait()
# optimization is completed, print the top performing experiments id
top_exp = an_optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])
# make sure background optimization stopped
an_optimizer.stop()

print('We are done, good bye')