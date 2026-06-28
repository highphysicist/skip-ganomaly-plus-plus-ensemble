##
import importlib

##
def load_model(opt, dataloader):
    """ Load model based on the model name.

    Arguments:
        opt {[argparse.Namespace]} -- options
        dataloader {[dict]} -- dataloader class

    Returns:
        [model] -- Returned model
    """
    model_name = opt.model
    if model_name == 'skipganomaly':
+        from lib.models.skipganomaly import Skipganomaly
+        return Skipganomaly(opt, data)
+    if model_name == 'skipganomaly_pp':
+        from lib.models.skipganomaly_pp import SkipganomalyPP
+        return SkipganomalyPP(opt, data)
+    if model_name == 'ensemble':
+        from lib.models.ensemble import SkipGanomalyEnsemble
+        return SkipGanomalyEnsemble(opt, data)
+    raise ValueError(f'Unknown model {model_name}')
    return model(opt, dataloader)
