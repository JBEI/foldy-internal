import axiosInstance from "../services/axiosInstance";
import { Logit } from "../types/types";

/**
 * Starts logits computation
 */
export const startLogits = async (foldId: number, name: string, logitModel: string, useStructure: boolean, getDepthTwoLogits: boolean): Promise<Logit> => {
    const response = await axiosInstance.post(
        `/api/startlogits/${foldId}`, {
        name: name,
        logit_model: logitModel,
        use_structure: useStructure,
        get_depth_two_logits: getDepthTwoLogits,
    });
    return response.data;
};

/**
 * Starts embeddings computation
 */
export const startEmbeddings = async (
    foldId: number,
    batchName: string,
    dmsStartingSeqIds: string[],
    extraSeqIds: string[],
    extraLayers: string[],
    embeddingModel: string
): Promise<boolean> => {
    const response = await axiosInstance.post(
        `/api/embeddings/${foldId}`,
        {
            batch_name: batchName,
            embedding_model: embeddingModel,
            dms_starting_seq_ids: dmsStartingSeqIds,
            extra_seq_ids: extraSeqIds,
            extra_layers: extraLayers,
        }
    );
    return response.data;
};
