import dataUtils


discrete_obs_dataframe = dataUtils.loadSachsObservationalDiscrete()
discrete_obs_data = discrete_obs_dataframe.values
discrete_obs_data_names = list(discrete_obs_dataframe.columns)
# print(discrete_obs_dataframe)
# print(discrete_obs_data)
# print(discrete_obs_data_names)

log_obs_df, continuous_log_df = dataUtils.loadSachsObservational()
continuous_names = list(log_obs_df.columns)
log_obs_data = log_obs_df.values
continuous_obs_data = continuous_log_df.values
print(log_obs_df)
print(continuous_log_df)
