from nbconvert.preprocessors import Preprocessor


# Some packages use log messages with carriage returns (\r) to
# overwrite previous logs and give progress updates.
# That usually renders fine in jupyterlab, but in some places, all
# the intermediate lines are displayed. This preprocessor removes any
# text before a carriage return
class RemoveCarriageReturnLines(Preprocessor):
    def preprocess_cell(self, cell, resources, cell_index):
        """Preprocess a single cell."""
        if cell.cell_type == 'code':
            for output in cell.outputs:
                if output.output_type == "stream":
                    output.text = output.text.split("\r")[-1]
        return cell, resources


c = get_config()

c.NbConvertApp.notebooks = ["notebooks/sbi.ipynb"]
c.ExecutePreprocessor.enabled = True
c.NotebookExporter.preprocessors = [RemoveCarriageReturnLines]
c.Application.log_level = "DEBUG"
c.NbConvertApp.use_output_suffix=False
c.NbConvertApp.export_format='notebook'
c.FilesWriter.build_directory=''
