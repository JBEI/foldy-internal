import { BoltzYamlHelper } from "../util/boltzYamlHelper";

export interface FoldInput {
    name: string;
    tags: string[];

    yaml_config: string | null;
    diffusion_samples: number | null;
    yaml_helper: BoltzYamlHelper | null;

    // Old inputs.
    sequence: string | null;
    af2_model_preset: string | null;
    disable_relaxation: boolean | null;
}

export interface Fold extends FoldInput {
    id: number | null;
    owner: string;
    create_date: string; // ISO formatted datetime string
    public: boolean | null;
    state: string | null;
    jobs: Invokation[] | null;
    docks: Dock[] | null;
    logits: Logit[] | null;
    embeddings: Embedding[] | null;
    evolutions: Evolution[] | null;
}

export interface FoldPae {
    pae: number[][];
}

export interface FoldContactProb {
    contact_prob: number[][];
}


export interface AffinityPrediction {
    affinity_pred_value: number;             // Predicted binding affinity from the ensemble model
    affinity_probability_binary: number;      // Predicted binding likelihood from the ensemble model
    affinity_pred_value1: number;            // Predicted binding affinity from the first model
    affinity_probability_binary1: number;     // Predicted binding likelihood from first model
    affinity_pred_value2: number;            // Predicted binding affinity from the second model
    affinity_probability_binary2: number;     // Predicted binding likelihood from second model
}


export const getJobStatus = (fold: Fold, job_type: string): string | null => {
    if (!fold.jobs) {
        return null;
    }
    for (const job of fold.jobs) {
        if (job.type === job_type) {
            return job.state;
        }
    }
    return null;
};


export const foldIsFinished = (fold: Fold): boolean => {
    return getJobStatus(fold, "models") === "finished";
};

export interface DockInput {
    fold_id: number;
    ligand_name: string;
    ligand_smiles: string;
    tool: string | null;
    bounding_box_residue: string | null;
    bounding_box_radius_angstrom: number | null;
}

export interface Dock extends DockInput {
    id: number;
    invokation_id: number | null;
    pose_energy: number | null;
    pose_confidences: string | null;
}

export interface Logit {
    id: number;
    name: string;
    fold_id: number;
    logit_model: string;
    use_structure: boolean | null;
    get_depth_two_logits: boolean | null;
    invokation_id: number;
}

export interface Embedding {
    id: number;
    name: string;
    fold_id: number;
    embedding_model: string;
    extra_seq_ids: string | null;
    dms_starting_seq_ids: string | null;
    homolog_fasta: string | null;
    extra_layers: string | null;
    invokation_id: number | null;
}

export interface Evolution {
    id: number | undefined;
    name: string;
    num_mutants: number;
    fold_id: number;
    invokation_id: number | undefined;
    mode: string;
    embedding_files: string | undefined;
    naturalness_files: string | undefined;
    finetuning_model_checkpoint: string | undefined;
    few_shot_params: string | undefined;
}

export interface Invokation {
    id: number;
    type: string | null;
    job_id: string | null;
    state: string | null;
    command: string | null;
    log: string | null;
    timedelta_sec: number | null;
    starttime: string | null; // ISO formatted datetime string
}

export interface Annotations {
    [chainName: string]: Array<{
        type: string;
        start: number;
        end: number;
    }>;
}

export interface RenderableAnnotation {
    type: string;
    start: number;
    end: number;
    color: string;
}

export interface RenderableAnnotations {
    [chainName: string]: Array<RenderableAnnotation>;
}

export interface FileInfo {
    key: string;
    size: number;
    modified: number;
}






///////////////////


// // ----- Interfaces -----
// export interface DockInput {
//     fold_id: number;
//     ligand_name: string;
//     ligand_smiles: string;
//     tool: string | null;
//     bounding_box_residue: string | null;
//     bounding_box_radius_angstrom: number | null;
// }

// export interface Dock extends DockInput {
//     id: number;
//     invokation_id: number | null;
//     pose_energy: number | null;
//     pose_confidences: string | null;
// }

// export interface FoldInput {
//     name: string;
//     tags: string[];
//     sequence: string;
//     af2_model_preset: string | null;
//     disable_relaxation: boolean | null;
// }

// export interface Invokation {
//     id: number;
//     type: string | null;
//     job_id: string | null;
//     state: string | null;
//     command: string | null;
//     log: string | null;
//     timedelta_sec: number | null;
//     starttime: string | null; // ISO formatted datetime string.
// }

// export interface Fold extends FoldInput {
//     id: number | null;
//     owner: string;
//     create_date: string; // ISO formatted datetime string.
//     public: boolean | null;
//     state: string | null;
//     jobs: Invokation[] | null;
//     docks: Dock[] | null;
// }

// export interface FoldPae {
//     pae: number[][];
// }

// export interface FoldContactProb {
//     contact_prob: number[][];
// }

// export interface Annotations {
//     [chainName: string]: [
//         {
//             type: string;
//             start: number;
//             end: number;
//         }
//     ];
// }

// export interface FileInfo {
//     key: string;
//     size: number;
//     modified: number;
// }
