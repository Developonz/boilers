"""Fresh native warm start without deepcopy or cached training-pool state."""
from pathlib import Path
from tempfile import TemporaryDirectory


def fit_continued(estimator,features,targets,previous=None,**kwargs):
    if previous is None:
        return estimator.fit(features,targets,init_model=None,**kwargs)
    # Passing a just-trained CatBoost object may retain training-pool context.
    # The documented file warm start carries only the fitted ensemble. Loading
    # through init_model's file API also avoids Python __deepcopy__/deserialize.
    with TemporaryDirectory(prefix='boiler-catboost-init-') as folder:
        path=Path(folder)/'previous.cbm'
        previous.save_model(str(path),format='cbm')
        return estimator.fit(features,targets,init_model=str(path),**kwargs)
