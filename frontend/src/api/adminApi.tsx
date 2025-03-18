import axiosInstance from '../services/axiosInstance';

/**
 * Creates database tables
 */
export const createDbs = async (): Promise<any> => {
  const response = await axiosInstance.post('/api/createdbs', {});
  return response.data;
};

/**
 * Upgrades database tables
 */
export const upgradeDbs = async (): Promise<any> => {
  const response = await axiosInstance.post('/api/upgradedbs', {});
  return response.data;
};

/**
 * Stamps database with revision
 */
export const stampDbs = async (revision: string): Promise<any> => {
  const response = await axiosInstance.post('/api/stampdbs', { revision });
  return response.data;
};

/**
 * Queues a test job
 */
export const queueTestJob = async (queue: string): Promise<any> => {
  const response = await axiosInstance.post('/api/queuetestjob', { queue });
  return response.data;
};

/**
 * Removes failed jobs from a queue
 */
export const removeFailedJobs = async (queue: string): Promise<any> => {
  const response = await axiosInstance.post('/api/remove_failed_jobs', { queue });
  return response.data;
};

/**
 * Kills a worker
 */
export const killWorker = async (workerToKill: string): Promise<any> => {
  const response = await axiosInstance.post(
    '/api/kill_worker',
    { worker_id: workerToKill }
  );
  return response.data;
};

/**
 * Sends a test email
 */
export const sendTestEmail = async (): Promise<any> => {
  const response = await axiosInstance.post('/api/sendtestemail', {});
  return response.data;
};

/**
 * Adds invokation to all jobs of a specific type and state
 */
export const addInvokationToAllJobs = async (
  jobType: string,
  jobState: string
): Promise<any> => {
  const response = await axiosInstance.post(
    `/api/addInvokationToAllJobs/${jobType}/${jobState}`,
    {}
  );
  return response.data;
};

/**
 * Runs unrun stages of a specific type
 */
export const runUnrunStages = async (stageToRun: string): Promise<any> => {
  const response = await axiosInstance.post(
    `/api/runUnrunStages/${stageToRun}`,
    {}
  );
  return response.data;
};

/**
 * Sets all unset model presets
 */
export const setAllUnsetModelPresets = async (): Promise<boolean> => {
  const response = await axiosInstance.post(
    '/api/set_all_unset_model_presets',
    {}
  );
  return response.data;
};

/**
 * Kills folds in a range
 */
export const killFoldsInRange = async (foldRange: string): Promise<any> => {
  const response = await axiosInstance.post(
    `/api/killFolds/${foldRange}`,
    {}
  );
  return response.data;
};

/**
 * Bulk adds a tag to folds in a range
 */
export const bulkAddTag = async (foldRange: string, newTag: string): Promise<any> => {
  const response = await axiosInstance.post(
    `/api/bulkAddTag/${foldRange}/${newTag}`,
    {}
  );
  return response.data;
};