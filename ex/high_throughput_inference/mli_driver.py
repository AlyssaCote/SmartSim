import os
import sys
from smartsim import Experiment
from smartsim.status import TERMINAL_STATUSES
import time

device = "gpu"
filedir = os.path.dirname(__file__)
worker_manager_script_name = os.path.join(filedir, "standalone_workermanager.py")
app_script_name = os.path.join(filedir, "mock_app.py")
model_name = os.path.join(filedir, f"resnet50.{device.upper()}.pt")


exp = Experiment("MLI_proto", launcher="dragon", exp_path=os.path.join(filedir, "MLI_proto"))

worker_manager_rs = exp.create_run_settings(sys.executable, [worker_manager_script_name, "--device", device])
worker_manager = exp.create_model("worker_manager", run_settings=worker_manager_rs)
worker_manager.attach_generator_files(to_copy=[worker_manager_script_name])


app_rs = exp.create_run_settings(sys.executable, exe_args = [app_script_name, "--device", device])
app = exp.create_model("app", run_settings=app_rs)
app.attach_generator_files(to_copy=[app_script_name], to_symlink=[model_name])


exp.generate(worker_manager, app, overwrite=True)
exp.start(worker_manager, app, block=False)

while True:
    if exp.get_status(app)[0] in TERMINAL_STATUSES:
        exp.stop(worker_manager)
        break
    if exp.get_status(worker_manager)[0] in TERMINAL_STATUSES:
        exp.stop(app)
        break
    time.sleep(5)

print("Exiting.")