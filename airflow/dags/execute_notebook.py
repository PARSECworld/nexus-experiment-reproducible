
import papermill as pm
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

class ExecuteNotebookOperator(BaseOperator):

    @apply_defaults
    def __init__(self, notebook_path, output_path, parameters=None, *args, **kwargs):
        super(ExecuteNotebookOperator, self).__init__(*args, **kwargs)
        self.notebook_path = notebook_path
        self.output_path = output_path
        self.parameters = parameters or {}

    def execute(self, context):
        pm.execute_notebook(
            self.notebook_path,
            self.output_path,
            parameters=self.parameters
        )
