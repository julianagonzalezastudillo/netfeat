""" This file contains auxiliary functions """
import save_features


def add_attributes(obj, **kwargs):
    """
    Adds attributes to an object dynamically. Created to add necessary attributes to pipelines.
    """
    for key, value in kwargs.items():
        setattr(obj, key, value)


def save_global(self):
    selec = {
        "dataset": self.dataset.code,
        "subject": self.subject,
        "session": self.session,
        "pipeline": self.pipeline,
        "metric": self.metric,
        "cv": save_features.n_cv,  # n_cv,
        "ch": self.subelec_names,
        "t-val": self.t_val,
    }
    save_features.df_select = save_features.df_select.append(selec, ignore_index=True)
    if self.subject == self.dataset.subject_list[-1] \
            and save_features.n_cv == self.cv_splits\
            and self.session == self.sessions_name[-1]:
        df_save = save_features.df_select[(save_features.df_select["dataset"] == self.dataset.code) &
                                          (save_features.df_select["pipeline"] == self.pipeline)]
        df_save.to_csv("./results/select_features/select_features_{0}_{1}.csv"
                       .format(self.dataset.code, self.pipeline), index=False)
    save_features.n_cv = save_features.n_cv + 1
