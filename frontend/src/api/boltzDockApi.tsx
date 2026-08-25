import axiosInstance from '../services/axiosInstance';

export interface BoltzDockLigandInput {
    name: string;
    smiles: string;
    chain_id?: string;
}

export interface BoltzDockStateInput {
    name: string;
    role: 'pre' | 'post' | 'ligand';
    components: BoltzDockLigandInput[];
}

export interface BoltzDockComparisonInput {
    name: string;
    pre_state: string;
    post_state: string;
}

export interface BoltzDockCofactorInput {
    chain_id: string;
    ccd: string;
}

export interface BoltzDockBatchInput {
    name: string;
    source_fold_id: number;
    campaign_round_id?: number;
    variants: Array<string | { seq_id: string; sequence: string }>;
    ligands?: BoltzDockLigandInput[];
    states?: BoltzDockStateInput[];
    comparisons?: BoltzDockComparisonInput[];
    protein_chain_id?: string;
    ligand_chain_id?: string;
    diffusion_samples: number;
    msa_mode: 'server' | 'reuse_source';
    cofactors?: BoltzDockCofactorInput[];
    bonds?: Array<{
        atom1: [string, number, string];
        atom2: [string, number, string];
    }>;
    pocket?: {
        contacts: Array<[string, string | number]>;
        max_distance: number;
        force: boolean;
    };
    activities?: Array<{ seq_id: string; activity: number }>;
    tags?: string[];
    start_jobs?: boolean;
}

export interface BoltzDockScoreData {
    best_model?: number;
    model_count?: number;
    grading_error?: string;
    confidence_score?: number | null;
    ptm?: number | null;
    iptm?: number | null;
    ligand_iptm?: number | null;
    complex_plddt?: number | null;
    complex_iplddt?: number | null;
    ligand_plddt?: number | null;
    anchor_distance?: number | null;
    target_distance?: number | null;
    closest_target_distance?: number | null;
    pose_rmsd?: number | null;
    delta_ligand_iptm_vs_wt?: number | null;
    warnings?: string[];
    components?: Array<{
        name: string;
        chain_id: string;
        plddt?: number | null;
        target_distance?: number | null;
    }>;
}

export interface BoltzDockEntry {
    id: number;
    fold_id: number;
    fold_name: string;
    seq_id: string;
    ligand_name: string;
    ligand_smiles: string;
    state_data?: BoltzDockStateInput | null;
    state: string;
    setup_error?: string | null;
    graded_at?: string | null;
    activity?: number | null;
    pose_quality_rank?: number | null;
    score_data?: BoltzDockScoreData | null;
}

export interface BoltzDockComparisonResult {
    comparison_name: string;
    seq_id: string;
    pre_state: string;
    post_state: string;
    pre_fold_id?: number;
    post_fold_id?: number;
    maintained_atom_rmsd?: number | null;
    mean_maintained_atom_displacement?: number | null;
    max_maintained_atom_displacement?: number | null;
    mapped_atom_count?: number;
    pre_heavy_atom_count?: number;
    post_heavy_atom_count?: number;
    maintained_atom_fraction?: number;
    lost_atom_count?: number;
    lost_atoms?: Array<{
        component: string;
        atoms: Array<{ atom_index: number; heavy_atom_index: number; element: string }>;
    }>;
    comparison_error?: string;
}

export interface BoltzDockBatch {
    id: number;
    name: string;
    source_fold_id: number;
    campaign_round_id?: number | null;
    created_at: string;
    config: Record<string, unknown>;
    entry_count: number;
    state_counts: Record<string, number>;
    comparison_data?: {
        definitions: Array<Record<string, unknown>>;
        results: BoltzDockComparisonResult[];
    } | null;
    state_summaries?: Record<string, {
        graded_count: number;
        metric_correlations: Record<string, { spearman?: number | null; n: number }>;
    }>;
    ligand_summaries?: Record<string, {
        graded_count: number;
        metric_correlations: Record<string, { spearman?: number | null; n: number }>;
    }>;
    comparison_summaries?: Record<string, {
        graded_count: number;
        maintained_atom_rmsd_activity_spearman?: number | null;
        activity_pair_count: number;
    }>;
    entries?: BoltzDockEntry[];
    graded_entries?: number;
}

export const getBoltzDockBatches = async (params: {
    sourceFoldId?: number;
    campaignRoundId?: number;
}): Promise<BoltzDockBatch[]> => {
    const response = await axiosInstance.get<{ batches: BoltzDockBatch[] }>(
        '/api/boltz_dock_batches',
        {
            params: {
                source_fold_id: params.sourceFoldId,
                campaign_round_id: params.campaignRoundId,
            },
        }
    );
    return response.data.batches;
};

export const getBoltzDockBatch = async (batchId: number): Promise<BoltzDockBatch> => {
    const response = await axiosInstance.get<BoltzDockBatch>(
        `/api/boltz_dock_batches/${batchId}`
    );
    return response.data;
};

export const createBoltzDockBatch = async (
    input: BoltzDockBatchInput
): Promise<BoltzDockBatch> => {
    const response = await axiosInstance.post<BoltzDockBatch>(
        '/api/boltz_dock_batches',
        input
    );
    return response.data;
};

export const gradeBoltzDockBatch = async (batchId: number): Promise<BoltzDockBatch> => {
    const response = await axiosInstance.post<BoltzDockBatch>(
        `/api/boltz_dock_batches/${batchId}/grade`
    );
    return response.data;
};
