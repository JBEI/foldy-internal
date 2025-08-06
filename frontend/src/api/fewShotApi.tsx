import { authHeader } from '../util/authHeader';
import axiosInstance from '../services/axiosInstance';
import { FewShot } from '../types/types';
import { getFile } from './fileApi';

export const getFewShot = async (fewShotId: number): Promise<FewShot> => {
    const response = await axiosInstance.get<FewShot>(`/api/few_shots/${fewShotId}`);
    return response.data;
};

export async function runFewShot(
    fewShotName: string,
    foldId: number,
    activityFile: File | null,
    activityFilePath: string | null,
    activityFileFromFewShotId: number | null,
    activityFileFromCampaignRoundId: number | null,
    mode: string,
    numMutants: number,
    embeddingFiles?: string[],
    naturalnessFiles?: string[],
    finetuningModelCheckpoint?: string,
    fewShotParams?: string,
): Promise<FewShot> {
    const formData = new FormData();
    formData.append('name', fewShotName);
    formData.append('fold_id', foldId.toString());
    if (activityFile) {
        formData.append('activity_file_bytes', activityFile);
    }
    if (activityFilePath) {
        formData.append('activity_file_path', activityFilePath);
    }
    if (activityFileFromFewShotId) {
        formData.append('activity_file_from_few_shot_id', activityFileFromFewShotId.toString());
    }
    if (activityFileFromCampaignRoundId) {
        formData.append('activity_file_from_campaign_round_id', activityFileFromCampaignRoundId.toString());
    }

    const createDirResponse = await axiosInstance.post<FewShot>('/api/few_shots/create_few_shot_directory', formData, {
        headers: {
            ...authHeader(),  // Keep auth headers
            'Content-Type': 'multipart/form-data',  // Override content type for this request
        }
    });
    if (createDirResponse.status !== 200) {
        throw new Error(`Failed to create few shot directory: ${createDirResponse.statusText}`);
    }

    const startFewShotBody: any = {
        name: fewShotName,
        fold_id: foldId,
        mode: mode,
        num_mutants: numMutants,
        embedding_files: embeddingFiles?.join(',') ?? undefined,
        naturalness_files: naturalnessFiles?.join(',') ?? undefined,
        finetuning_model_checkpoint: finetuningModelCheckpoint ?? undefined,
        few_shot_params: fewShotParams ?? undefined,
    };
    // if (fewShotParams) {
    //     startFewShotBody.few_shot_params = JSON.stringify(fewShotParams);
    // }

    const response = await axiosInstance.post<FewShot>(
        `/api/few_shot`, startFewShotBody,
        // {
        //     headers: {
        //         'Content-Type': 'application/json',
        //     },
        // }
    );
    return response.data;
}


export const deleteFewShot = async (fewShotId: number): Promise<void> => {
    await axiosInstance.delete(`/api/few_shots/${fewShotId}`);
};

/**
 * Interface for FewShot debug information (training metrics only)
 */
export interface FewShotDebugInfo {
    debugData: any | null;
    sortOptions: { [key: string]: string[] } | null;
}

/**
 * Load FewShot debug information (debug_info.json only) from a FewShot run
 * Uses the correct dynamic path resolution from output_fpath
 *
 * @param foldId - The fold ID
 * @param fewShotRun - The FewShot run object
 * @returns Promise containing debug data and sort options
 */
export const getFewShotDebugInfo = async (
    foldId: number,
    fewShotRun: FewShot
): Promise<FewShotDebugInfo> => {
    const result: FewShotDebugInfo = {
        debugData: null,
        sortOptions: null
    };

    try {
        // Load debug data using dynamic path resolution (the correct approach)
        if (fewShotRun.output_fpath) {
            const debugPath = fewShotRun.output_fpath.split('/').slice(0, -1).join('/') + '/debug_info.json';
            try {
                const debugBlob = await getFile(foldId, debugPath);
                const debugText = await debugBlob.text();
                // Replace NaN with null for proper JSON parsing
                const cleanedString = debugText.replace(/NaN/g, 'null');
                const jsonData = JSON.parse(cleanedString);
                result.debugData = jsonData;
                result.sortOptions = jsonData.sorts || null;
            } catch (debugError) {
                console.warn('Debug data not available:', debugError);
            }
        }
    } catch (error) {
        console.error('Error loading FewShot debug info:', error);
        throw error;
    }

    return result;
};
