c = get_config()

c.NbConvertApp.notebooks = ["notebooks/sbi.ipynb"]
c.ExecutePreprocessor.enabled = True
c.Application.log_level = "DEBUG"
c.NbConvertApp.use_output_suffix=False
c.NbConvertApp.export_format='notebook'
c.FilesWriter.build_directory=''
