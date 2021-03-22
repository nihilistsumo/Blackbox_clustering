from clearml import Task
from clearml.automation.optuna.optuna import OptimizerOptuna
from clearml.automation.optimization import HyperParameterOptimizer
from clearml.automation.parameters import UniformIntegerParameterRange, DiscreteParameterRange, UniformParameterRange
import argparse

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

def run_hyperparam_optim(project_name, task_name, lambda_min, lambda_max, lambda_step, reg_min, reg_max, reg_step, check_exp_period):
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
            project_name=project_name, task_name=task_name).id

    an_optimizer = HyperParameterOptimizer(
        # This is the experiment we want to optimize
        base_task_id=args['template_task_id'],
        hyper_parameters=[
        UniformParameterRange('lambda_val', min_value=lambda_min, max_value=lambda_max, step_size=lambda_step),
        UniformParameterRange('reg', min_value=reg_min, max_value=reg_max, step_size=reg_step)
        ],
        objective_metric_title='val_ARI',
        objective_metric_series='val_ARI',
        objective_metric_sign='max',
        max_number_of_concurrent_tasks=4,
        optimizer_class=search_strategy,
        execution_queue='default',
        # Optional: Limit the execution time of a single experiment, in minutes.
        # (this is optional, and if using  OptimizerBOHB, it is ignored)
        time_limit_per_job=60.,
        # Check the experiments every 6 seconds is way too often, we should probably set it to 5 min,
        # assuming a single experiment is usually hours...
        pool_period_min=check_exp_period,
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
    an_optimizer.set_report_period(check_exp_period)
    # start the optimization process, callback function to be called every time an experiment is completed
    # this function returns immediately
    an_optimizer.start(job_complete_callback=job_complete_callback)

    # set the time limit for the optimization process
    an_optimizer.set_time_limit(in_minutes=1440.0)
    # wait until process is done (notice we are controlling the optimization process in the background)
    an_optimizer.wait()
    # optimization is completed, print the top performing experiments id
    top_exp = an_optimizer.get_top_experiments(top_k=5)
    print([t.id for t in top_exp])
    # make sure background optimization stopped
    an_optimizer.stop()

    print('We are done, good bye')

def main():
    parser = argparse.ArgumentParser(description='Run hyper parameter optimization for BB clustering')
    parser.add_argument('-pn', '--project_name', default='BB Clustering')
    parser.add_argument('-tn', '--task_name', default='bbclustering_fixed_lambda')
    parser.add_argument('-lm', '--lambda_min', type=float, default=20.0)
    parser.add_argument('-lx', '--lambda_max', type=float, default=200.0)
    parser.add_argument('-ls', '--lambda_step', type=float, default=5.0)
    parser.add_argument('-rm', '--reg_min', type=float, default=0.0)
    parser.add_argument('-rx', '--reg_max', type=float, default=10.0)
    parser.add_argument('-rs', '--reg_step', type=float, default=0.2)
    parser.add_argument('-ep', '--report_period', type=float, default=1.0)
    args = parser.parse_args()
    project_name = args.project_name
    task_name = args.task_name
    lmin = args.lambda_min
    lmax = args.lambda_max
    lstep = args.lambda_step
    rmin = args.reg_min
    rmax = args.reg_max
    rstep = args.reg_step
    report_period = args.report_period

    run_hyperparam_optim(project_name, task_name, lmin, lmax, lstep, rmin, rmax, rstep, report_period)

if __name__ == '__main__':
    main()