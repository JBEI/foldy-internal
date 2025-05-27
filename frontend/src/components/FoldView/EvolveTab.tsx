import React, { useState, ChangeEvent, useMemo, useEffect } from 'react';
import UIkit from 'uikit';
import { FileInfo, Evolution, Invokation } from 'src/types/types';
import { deleteEvolution, evolve } from '../../api/evolveApi';
import { FaDownload, FaEye, FaFileCode, FaRedo, FaTrash } from 'react-icons/fa';
import fileDownload from 'js-file-download';
import { removeLeadingSlash } from '../../api/commonApi';
import { getFile } from '../../api/fileApi';
import { notify } from '../../services/NotificationService';
import Papa from 'papaparse';
import ReactDataGrid from 'react-data-grid';
import { BoltzYamlHelper } from '../../util/boltzYamlHelper';
import { Selection } from './StructurePane';
import Plot from 'react-plotly.js';
import { TabContainer, DescriptionSection, TableSection, CollapsibleSection, FormRow, FormField, ButtonGroup, ResponsiveTable } from '../../util/tabComponents';
import { TextInputControl, TextAreaControl, SelectControl, FileUploadControl, MultiSelectControl, NumberInputControl } from '../../util/controlComponents';
import { DataTableContainer, PlotContainer } from '../../util/plotComponents';
import { Row, Col, Form, Input, Select, Upload, Button as AntButton, Card, Divider, InputNumber, Alert, Modal, Typography } from 'antd';
import { UploadOutlined, PlayCircleOutlined, InfoCircleOutlined, QuestionCircleOutlined } from '@ant-design/icons';

const { Text, Paragraph, Title } = Typography;

const FEW_SHOT_PRESETS = {
    'folde_default_mlp': {
        mode: 'TorchMLPFewShotModel',
        params: `{
    "pretrain": true,
    "pretrain_epochs": 50,
    "ensemble_size": 5,
    "decision_mode": "ucb",
    "embedding_dim": 960,
    "hidden_dims": [100, 50],
    "dropout": 0.2,
    "learning_rate": 0.0003,
    "weight_decay": 0.00001,
    "train_epochs": 200,
    "train_patience": 40,
    "val_frequency": 10,
    "do_validation_with_pair_fraction": 0.2,
    "decision_mode": "constantliar",
    "lie_noise_stddev_multiplier": 4.0
}`
    },
    'evolvepro': {
        mode: 'RandomForestFewShotModel',
        params: `{
    "n_estimators": 100,
    "criterion": "friedman_mse",
    "max_depth": null,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 1.0,
    "max_leaf_nodes": null,
    "min_impurity_decrease": 0.0,
    "bootstrap": true,
    "oob_score": false,
    "n_jobs": null,
    "random_state": 1,
    "verbose": 0,
    "warm_start": false,
    "ccp_alpha": 0.0,
    "max_samples": null
}`
    },
    'custom': {
        mode: null,
        params: ''
    }
};

type RowData = {
    seqId: string;
    selectedIdx: number | null;
    relevantMeasuredMutants: string;
    predictionMean: number;
    predictionStddev: number;
    score: number;
    modelPredictions?: number[];
}


const parseCsvDataIntoRowData = (predictedMutantCsvDataString: string): RowData[] | null => {
    const { data, errors } = Papa.parse<Record<string, string>>(predictedMutantCsvDataString, {
        header: true,
        delimiter: ',',
        skipEmptyLines: true,
        dynamicTyping: true
    });

    if (errors.length > 0) {
        notify.error(`Error parsing predicted mutant CSV: ${errors.map(error => error.message).join(', ')}`);
        return null;
    }

    const allRows = data.map((row) => {
        // Iterate over columns from model_0 upward until none is found, adding scores to a list.
        const predictions: number[] = [];
        for (let i = 0; i < 100; i++) {
            const predictionStr = row[`model_${i}`];
            if (predictionStr) {
                // Try converting to number.
                const prediction = parseFloat(predictionStr);
                if (!isNaN(prediction)) {
                    predictions.push(prediction);
                }
            }
        }
        // Compute score mean and stddev.
        const mean = predictions.reduce((a, b) => a + b, 0.0) / predictions.length;
        const stddev = Math.sqrt(predictions.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / predictions.length);
        const score = mean;

        // Parse selected_idx from the CSV
        const selectedIdxStr = row['selected_idx'];
        let selectedIdx: number | null = null;
        if (selectedIdxStr && selectedIdxStr !== 'null' && selectedIdxStr !== '') {
            const parsed = parseInt(selectedIdxStr);
            if (!isNaN(parsed)) {
                selectedIdx = parsed;
            }
        }

        return {
            seqId: row['seq_id'],
            relevantMeasuredMutants: row['relevant_measured_mutants'],
            selectedIdx: selectedIdx,
            predictionMean: mean,
            predictionStddev: stddev,
            score: score,
            modelPredictions: predictions,
        };
    });

    // Filter to only include rows with selected_idx set, then sort by selected_idx
    const selectedRows = allRows.filter(row => row.selectedIdx !== null);
    return selectedRows.sort((a, b) => a.selectedIdx! - b.selectedIdx!);
}

const seqIdListToLociList = (seqIdList: string[]): number[] => {
    const lociToHighlightList: number[] = [];
    seqIdList.forEach(seqId => {
        const loci = seqId.split('_').map((alleleId) => {
            // alleleId[1:-1];
            const locusStr = alleleId.slice(1, -1);
            const locusInt = parseInt(locusStr);
            if (isNaN(locusInt)) {
                console.log(`Invalid locus ${locusStr} for seqId ${seqId}`);
                return null;
            }
            return locusInt;
        }).filter(locus => locus !== null);
        lociToHighlightList.push(...loci);
    })

    return Array.from(new Set(lociToHighlightList));
}




interface PredictedMutantTableProps {
    yamlConfig: string | null;
    predictedMutantCsvData: string | null;
    setSelectedSubsequence: (selection: Selection | null) => void;
}


// Now modify the PredictedMutantTable component to include the heatmap
const PredictedMutantTable: React.FC<PredictedMutantTableProps> = ({
    yamlConfig,
    predictedMutantCsvData,
    setSelectedSubsequence,
}) => {
    if (!predictedMutantCsvData) {
        return <div className="uk-text-center">
            <div uk-spinner="ratio: 4"></div>
        </div>;
    }

    const [sortColumn, setSortColumn] = useState<string | null>(null);
    const [sortDirection, setSortDirection] = useState<'ASC' | 'DESC'>('DESC');
    const [selectedSeqIds, setSelectedSeqIds] = useState<string[]>([]);

    const tableData: RowData[] | null = useMemo(() => {
        const data = parseCsvDataIntoRowData(predictedMutantCsvData);
        if (!data) return null;

        if (data && sortColumn) {
            return [...data].sort((a, b) => {
                const aValue = a[sortColumn as keyof RowData];
                const bValue = b[sortColumn as keyof RowData];
                if (aValue === null || aValue === undefined) return 1;
                if (bValue === null || bValue === undefined) return -1;
                return sortDirection === 'ASC'
                    ? (aValue < bValue ? -1 : 1)
                    : (aValue > bValue ? -1 : 1);
            });
        }
        return data;
    }, [predictedMutantCsvData, sortColumn, sortDirection]);

    const correlationData = useMemo(() => {
        if (!tableData || tableData.length === 0 || !tableData[0].modelPredictions) {
            return null;
        }

        const sequenceCount = tableData.length;
        if (sequenceCount <= 1) return null;

        // Create a matrix to store correlations between sequences
        const matrix: number[][] = Array(sequenceCount).fill(0).map(() => Array(sequenceCount).fill(0));

        // For each pair of sequences, we need to calculate correlation of their values
        // across the different metrics (mean, stddev, score)
        for (let i = 0; i < sequenceCount; i++) {
            for (let j = 0; j < sequenceCount; j++) {
                // For identical sequences, correlation is 1
                if (i === j) {
                    matrix[i][j] = 1;
                    continue;
                }

                // Calculate correlation coefficient between model predictions
                const seqIData = tableData[i].modelPredictions!;
                const seqJData = tableData[j].modelPredictions!;

                // Calculate means
                const iMean = seqIData.reduce((sum, val) => sum + val, 0) / seqIData.length;
                const jMean = seqJData.reduce((sum, val) => sum + val, 0) / seqJData.length;

                // Calculate correlation coefficient (normalized covariance)
                let numerator = 0;
                let iVariance = 0;
                let jVariance = 0;

                for (let k = 0; k < seqIData.length; k++) {
                    const iDiff = seqIData[k] - iMean;
                    const jDiff = seqJData[k] - jMean;
                    numerator += iDiff * jDiff;
                    iVariance += iDiff * iDiff;
                    jVariance += jDiff * jDiff;
                }

                // Correlation coefficient = covariance / (stddev_i * stddev_j)
                const correlation = numerator / (Math.sqrt(iVariance) * Math.sqrt(jVariance));

                matrix[i][j] = correlation;
            }
        }

        return matrix;
    }, [tableData]);

    if (!tableData) return null;
    const columns = [
        {
            key: "selectedIdx",
            name: "Selected",
            sortable: true,
            resizable: true,
            width: 80,
            sortDescendingFirst: false,
            formatter: ({ row }: { row: any }) => (
                <div style={{ textAlign: 'center' }}>
                    {row.selectedIdx !== null ? row.selectedIdx : ''}
                </div>
            )
        },
        {
            key: "seqId",
            name: "Sequence ID",
            sortable: true,
            resizable: true,
            // width: 200,
            sortDescendingFirst: true,
            formatter: ({ row }: { row: any }) => (
                <div
                    uk-tooltip={row.seqId}
                    style={{
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        paddingLeft: '5px'
                    }}
                >
                    {row.seqId}
                </div>
            )
        },
        {
            key: 'relevantMeasuredMutants',
            name: "Measured",
            resizable: true,
            // width: 200,
            formatter: ({ row }: { row: any }) => (
                <div
                    uk-tooltip={row.relevantMeasuredMutants}
                    style={{
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                    }}
                >
                    {row.relevantMeasuredMutants}
                </div>
            )
        },
        {
            key: 'predictionMean',
            name: "Mean",
            sortable: true,
            resizable: true,
            width: 70,
            formatter: ({ row }: { row: any }) => (
                <div uk-tooltip={row.predictionMean.toFixed(4)} style={{ textAlign: 'left' }}>
                    {row.predictionMean.toFixed(2)}
                </div>
            )
        },
        {
            key: "predictionStddev",
            name: "STD",
            sortable: true,
            resizable: true,
            width: 70,
            formatter: ({ row }: { row: any }) => (
                <div uk-tooltip={row.predictionStddev.toFixed(4)} style={{ textAlign: 'left' }}>
                    {row.predictionStddev.toFixed(2)}
                </div>
            )
        }
    ];

    // Existing functions
    const copyMutationsToClipboard = () => {
        if (!tableData) return;
        const mutations = tableData
            .map(row => row.seqId)
            .join('\n');

        navigator.clipboard.writeText(mutations);
        notify.success('Seq IDs copied to clipboard!');
    }

    const highlightResiduesOnModel = () => {
        if (!tableData) return null;
        if (!yamlConfig) {
            console.log('No yaml config, cannot highlight residues on model.');
            return;
        }
        const configHelper = new BoltzYamlHelper(yamlConfig);
        if (configHelper.getProteinSequences().length > 1) {
            notify.error('Cannot currently highlight residues on multimers.');
        }

        const uniqueLociToHighlight = seqIdListToLociList(tableData.map(row => row.seqId));

        const specialSelectedLoci = seqIdListToLociList(selectedSeqIds);

        const selection = uniqueLociToHighlight.map(locus => {
            const color = specialSelectedLoci.includes(locus) ? "#FFD700" : "#39f";
            return {
                struct_asym_id: 'A',
                start_residue_number: locus,
                end_residue_number: locus,
                color: color,
            }
        })

        setSelectedSubsequence({
            data: selection,
            nonSelectedColor: "white",
        });
    }

    useEffect(() => {
        highlightResiduesOnModel();
    }, [selectedSeqIds, tableData]);

    return (
        <DataTableContainer>
            <ReactDataGrid
                columns={columns}
                rowGetter={i => tableData[i]}
                rowsCount={tableData.length}
                onGridSort={(sortCol, direction) => {
                    setSortColumn(sortCol);
                    setSortDirection(direction.toUpperCase() as 'ASC' | 'DESC');
                }}
                onRowSelect={(rows) => {
                    setSelectedSeqIds(rows.map(row => row.seqId));
                }}
                onRowClick={(_, row) => {
                    setSelectedSeqIds([row.seqId]);
                }}
                minHeight={400}
            />
            <ButtonGroup>
                <button
                    className="uk-button uk-button-default"
                    onClick={() => copyMutationsToClipboard()}
                >
                    Copy mutations to clipboard
                </button>
                <button
                    className="uk-button uk-button-primary"
                    onClick={() => highlightResiduesOnModel()}
                >
                    Highlight residues on model
                </button>
            </ButtonGroup>

            {useMemo(() => {
                if (!correlationData) return null;

                return (
                    <div style={{
                        height: '600px', // Increased height for better visibility
                        backgroundColor: '#f9f9f9',
                        padding: '15px',
                        borderRadius: '4px',
                        marginTop: '20px',
                        overflowX: 'auto', // Add horizontal scroll for many sequences
                        overflowY: 'auto'  // Add vertical scroll too
                    }}>
                        <Plot
                            data={[{
                                z: correlationData,
                                x: tableData.map(row => row.seqId),
                                y: tableData.map(row => row.seqId),
                                type: 'heatmap',
                                colorscale: 'RdBu',
                                zmin: -1,
                                zmax: 1,
                                hovertemplate: '%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>',
                                showscale: true,
                                colorbar: {
                                    title: 'Correlation',
                                    titleside: 'right'
                                }
                            }]}
                            layout={{
                                title: 'Sequence Prediction Correlation',
                                autosize: true,
                                // Increase margins to accommodate sequence IDs
                                margin: { l: 150, r: 50, t: 60, b: 150 },
                                xaxis: {
                                    title: 'Sequence ID',
                                    tickangle: 45,
                                    tickfont: { size: 10 }
                                },
                                yaxis: {
                                    title: 'Sequence ID',
                                    autorange: 'reversed',
                                    tickfont: { size: 10 }
                                },
                                plot_bgcolor: '#f9f9f9',
                                paper_bgcolor: '#f9f9f9',
                                font: { family: 'Arial, sans-serif' }
                            }}
                            style={{ width: '100%', height: '100%' }}
                            useResizeHandler={true}
                            config={{
                                responsive: true,
                                displayModeBar: true,
                                displaylogo: false,
                                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                                toImageButtonOptions: {
                                    format: 'png',
                                    filename: 'sequence_correlation_heatmap',
                                    height: 800,
                                    width: 800,
                                    scale: 2
                                }
                            }}
                        />
                    </div>
                );
            }, [correlationData, tableData])}
        </DataTableContainer>
    );
};


const createPlotData = (debugData: any) => {
    if (!debugData || !debugData.pretrain_metrics || !debugData.finetune_metrics) {
        return {
            pretrain: [],
            finetune: []
        };
    }

    // Create pretrain traces - one for each model's train and val loss
    const pretrainData: any[] = [];
    const finetuneData: any[] = [];

    // Inside the pretrain metrics loop:
    debugData.pretrain_metrics.forEach((model: any, index: number) => {
        if (model.train_loss && model.train_loss.length > 0) {
            pretrainData.push({
                x: Array.from({ length: model.train_loss.length }, (_, i) => i + 1),
                y: model.train_loss,
                name: `Model ${index + 1} Train`,
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#3e7bfa', // Blue for all train
                    width: 2,
                    dash: 'solid'
                }
            });
        }

        if (model.val_loss && model.val_loss.length > 0) {
            pretrainData.push({
                x: Array.from({ length: model.val_loss.length }, (_, i) => i + 1),
                y: model.val_loss,
                name: `Model ${index + 1} Val`,
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#e91e63', // Pink for all val
                    width: 1,
                    dash: 'solid',
                    opacity: 0.5,
                }
            });
        }
    });

    // Inside the finetune metrics loop:
    debugData.finetune_metrics.forEach((model: any, index: number) => {
        if (model.train_loss && model.train_loss.some((val: number) => val !== 0 && val !== null)) {
            // Filter out zeros which appear to be placeholders
            const nonZeroTrainLoss = model.train_loss.map((val: number) => val === 0 ? null : val);

            finetuneData.push({
                x: Array.from({ length: model.train_loss.length }, (_, i) => i + 1),
                y: nonZeroTrainLoss,
                name: `Model ${index + 1} Train`,
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#3e7bfa', // Blue for all train
                    width: 1,
                    dash: 'solid',
                    opacity: 0.5,
                }
            });
        }

        if (model.val_loss && model.val_loss.some((val: number) => val !== 0 && val !== null)) {
            // Filter out zeros which appear to be placeholders
            const nonZeroValLoss = model.val_loss.map((val: number) => val === 0 ? null : val);

            finetuneData.push({
                x: Array.from({ length: model.val_loss.length }, (_, i) => i + 1),
                y: nonZeroValLoss,
                name: `Model ${index + 1} Val`,
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#e91e63', // Pink for all val
                    width: 1,
                    dash: 'solid',
                    opacity: 0.5,
                }
            });
        }
    });

    return {
        pretrain: pretrainData,
        finetune: finetuneData
    };
};

// 5. Create a memoized component to render the plotly charts
const DebugPlots: React.FC<{ debugData: any }> = React.memo(({ debugData }) => {
    const plotData = useMemo(() => createPlotData(debugData), [debugData]);

    const pretrainPlot = useMemo(() => (
        <Plot
            data={plotData.pretrain}
            layout={{
                title: 'Pretraining Loss',
                autosize: true,
                margin: { l: 60, r: 25, t: 60, b: 60 },
                xaxis: {
                    title: 'Epoch',
                    gridcolor: '#e1e1e1'
                },
                yaxis: {
                    title: 'Loss',
                    gridcolor: '#e1e1e1'
                },
                plot_bgcolor: '#f9f9f9',
                paper_bgcolor: '#f9f9f9',
                font: { family: 'Arial, sans-serif' }
            }}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler={true}
            config={{
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            }}
        />
    ), [plotData.pretrain]);

    const finetunePlot = useMemo(() => (
        <Plot
            data={plotData.finetune}
            layout={{
                title: 'Finetuning Loss',
                autosize: true,
                margin: { l: 60, r: 25, t: 60, b: 60 },
                xaxis: {
                    title: 'Iteration',
                    gridcolor: '#e1e1e1'
                },
                yaxis: {
                    title: 'Loss',
                    gridcolor: '#e1e1e1'
                },
                plot_bgcolor: '#f9f9f9',
                paper_bgcolor: '#f9f9f9',
                font: { family: 'Arial, sans-serif' }
            }}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler={true}
            config={{
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            }}
        />
    ), [plotData.finetune]);

    if (!debugData) return null;

    return (
        <PlotContainer title="Training Metrics" height="700px">
            {/* Pretraining loss plot */}
            <div style={{
                height: '300px',
                backgroundColor: '#f9f9f9',
                padding: '15px',
                borderRadius: '4px'
            }}>
                {pretrainPlot}
            </div>

            {/* Finetuning loss plot */}
            <div style={{
                height: '300px',
                backgroundColor: '#f9f9f9',
                padding: '15px',
                borderRadius: '4px'
            }}>
                {finetunePlot}
            </div>
        </PlotContainer>
    );
});

interface EvolveTabProps {
    foldId: number;
    yamlConfig: string | null;
    jobs: Invokation[] | null;
    files: FileInfo[] | null;
    evolutions: Evolution[] | null;
    openUpLogsForJob: (jobId?: number) => void;
    setSelectedSubsequence: (selection: Selection | null) => void;
}

const EvolveTab: React.FC<EvolveTabProps> = ({ foldId, yamlConfig, jobs, files, evolutions, openUpLogsForJob, setSelectedSubsequence }) => {
    const [evolutionName, setEvolutionName] = useState<string>('');
    const [showForm, setShowForm] = useState<boolean>(false);
    const [activityFile, setActivityFile] = useState<File | null>(null);
    const [mode, setMode] = useState<string>('TorchMLPFewShotModel');
    const [numMutants, setNumMutants] = useState<number>(24);
    const [selectedEmbeddingPaths, setSelectedEmbeddingPaths] = useState<string[]>([]);
    const [selectedNaturalnessPaths, setSelectedNaturalnessPaths] = useState<string[]>([]);
    const [finetuningModelCheckpoint, setFinetuningModelCheckpoint] = useState<string>('facebook/esm2_t6_8M_UR50D');
    const [fewShotParams, setFewShotParams] = useState<string>('');
    const [selectedPreset, setSelectedPreset] = useState<string>('custom');

    const [displayedEvolutionId, setDisplayedEvolutionId] = useState<number | null>(null);
    const [evolutionCsvData, setEvolutionCsvData] = useState<string | null>(null);
    const [evolutionDebugData, setEvolutionDebugData] = useState<any>(null);

    const [showHelpModal, setShowHelpModal] = useState<boolean>(false);

    const availableEmbeddingFiles = files?.filter(file =>
        file.key.includes('embed')
    ) || [];
    const availableNaturalnessFiles = files?.filter(file =>
        file.key.includes('naturalness')
    ) || [];

    const handleEvolve = async () => {
        if (!activityFile || (selectedEmbeddingPaths.length === 0) || (selectedNaturalnessPaths.length === 0)) {
            notify.warning('Please fill in all required fields');
            return;
        }

        try {
            notify.info('Starting evolution...');
            const foldEvolution = await evolve(
                evolutionName,
                foldId,
                activityFile,
                mode,
                numMutants,
                selectedEmbeddingPaths,
                selectedNaturalnessPaths,
                mode === 'finetuning' ? finetuningModelCheckpoint : undefined,
                fewShotParams
            );
            notify.success(`Evolution process started with id ${foldEvolution.id} and name ${foldEvolution.name}`);
        } catch (error) {
            notify.error(`Failed to start evolution process: ${error}`);
        }
    };

    const getEvolutionStatus = (evolution: Evolution): string => {
        const job = jobs?.find(job => job.id === evolution.invokation_id);
        return job?.state || 'Unknown';
    };

    const downloadPredictedActivity = (evolution: Evolution) => {
        const predictedActivityPath = `evolve/${evolution.name}/predicted_activity.csv`;
        console.log(`Downloading predicted activity for evolution ${evolution.id} at path ${predictedActivityPath}`);
        getFile(foldId, predictedActivityPath).then(
            (fileBlob: Blob) => {
                const newFname = `${evolution.name}_predicted_activity.csv`;
                notify.info(`Downloading ${predictedActivityPath} with file name ${newFname}!`);
                fileDownload(fileBlob, newFname);
            },
            (e) => {
                console.log(e);
                notify.error(e.toString());
            }
        );
    };

    const rerunEvolution = async (evolution: Evolution) => {
        notify.info(`Repopulating "New Evolution Run" with parameters from ${evolution.name}. Make sure to add the activity file, you can download the previous one from Files tab.`);
        setEvolutionName(evolution.name);
        setMode(evolution.mode);
        if (evolution.embedding_files) {
            setSelectedEmbeddingPaths(evolution.embedding_files.split(','));
        }
        if (evolution.naturalness_files) {
            setSelectedNaturalnessPaths(evolution.naturalness_files.split(','));
        }
        if (evolution.finetuning_model_checkpoint) {
            setFinetuningModelCheckpoint(evolution.finetuning_model_checkpoint);
        }
        if (evolution.few_shot_params) {
            setFewShotParams(evolution.few_shot_params);
        }
        setShowForm(true);
    };

    const loadEvolution = (evolutionId: number) => {
        const evolution = evolutions?.find(evolution => evolution.id === evolutionId);
        if (!evolution) {
            notify.error(`Evolution ${evolutionId} not found.`);
            return;
        }
        setDisplayedEvolutionId(evolutionId);
        console.log(`Loading evolution ${evolution.name}...`);

        // Fetch the predicted activity file
        getFile(foldId, `evolve/${evolution.name}/predicted_activity.csv`).then(
            (fileBlob: Blob) => {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const fileString = e.target?.result as string;
                    setEvolutionCsvData(fileString);
                };
                reader.readAsText(fileBlob);
            },
            (e) => {
                console.log(e);
                notify.error(`Error fetching predicted_activity.csv: ${e.toString()}`);
            }
        );

        // Fetch the debug.json file
        getFile(foldId, `evolve/${evolution.name}/debug_info.json`).then(
            (fileBlob: Blob) => {
                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        const fileString = e.target?.result as string;
                        // Replace NaN with null for proper JSON parsing
                        const cleanedString = fileString.replace(/NaN/g, 'null');
                        const jsonData = JSON.parse(cleanedString);
                        setEvolutionDebugData(jsonData);
                    } catch (err) {
                        console.error("Error parsing debug.json:", err);
                        notify.error(`Failed to parse debug.json: ${err}`);
                        console.error(e.target?.result);
                    }
                };
                reader.readAsText(fileBlob);
            },
            (e) => {
                console.log(e);
                notify.error(`Error fetching debug.json: ${e.toString()}`);
            }
        );
    };

    const deleteEvolutionHelper = async (evolutionId: number) => {
        await UIkit.modal.confirm('Are you sure you want to delete this evolution? This action is irreversible.');
        console.log(`Deleting evolution ${evolutionId}...`);
        deleteEvolution(evolutionId).then(
            () => {
                notify.success(`Evolution ${evolutionId} deleted.`);
            },
            (e) => {
                notify.error(e.toString());
            }
        )
    }

    // Add this helper function after the FEW_SHOT_PRESETS dictionary
    const isValidJson = (jsonString: string): boolean => {
        if (!jsonString.trim()) return true; // Empty string is considered valid
        try {
            JSON.parse(jsonString);
            return true;
        } catch {
            return false;
        }
    };

    // Add this after the isValidJson function and before the return statement:
    const jsonValidationStatus = useMemo(() => {
        if (fewShotParams.trim() === '') return '';
        return isValidJson(fewShotParams) ? 'success' : 'error';
    }, [fewShotParams]);

    return (
        <TabContainer>
            {/* Description Section */}
            <DescriptionSection title="Evolution Runs Overview">
                This section accepts measurements of protein activity and suggests a slate of mutants for the next round of screening.
            </DescriptionSection>

            {/* Evolution Runs Table */}
            <TableSection title="Evolution Runs">
                <ResponsiveTable>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Status</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {evolutions?.map(evolution => (
                            <tr key={evolution.id}>
                                <td style={{ overflowX: 'hidden' }}><p uk-tooltip={evolution.name}>{evolution.name}</p></td>
                                <td>{getEvolutionStatus(evolution)}</td>
                                <td style={{ width: '200px', paddingLeft: '2px', paddingRight: '2px' }}>

                                    <FaFileCode
                                        uk-tooltip="View logs"
                                        onClick={() => openUpLogsForJob(evolution.invokation_id || undefined)}
                                    />
                                    {
                                        getEvolutionStatus(evolution) == 'finished' ?
                                            <>
                                                <FaEye
                                                    uk-tooltip="View results"
                                                    onClick={() => loadEvolution(evolution.id)} />
                                                <FaDownload
                                                    uk-tooltip="Download predicted activity CSV."
                                                    onClick={() => downloadPredictedActivity(evolution)} />
                                            </> :
                                            null
                                    }
                                    <FaRedo uk-tooltip="Retry the evolution run."
                                        onClick={() => rerunEvolution(evolution)} />
                                    <FaTrash
                                        uk-tooltip="Delete evolution run."
                                        onClick={() => deleteEvolutionHelper(evolution.id)} />
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </ResponsiveTable>
            </TableSection>


            {
                displayedEvolutionId ?
                    <TableSection title={""} scrollable={false}>
                        <div style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            marginBottom: "10px"
                        }}>
                            <h3 style={{ margin: 0, overflowWrap: 'anywhere' }}>
                                {evolutions?.find(e => e.id === displayedEvolutionId)?.name || "Evolution Results"}
                            </h3>
                            <button
                                onClick={() => setDisplayedEvolutionId(null)}
                                style={{
                                    background: "none",
                                    border: "none",
                                    cursor: "pointer",
                                    fontSize: "20px",
                                    padding: "5px",
                                    color: "#666"
                                }}
                                aria-label="Close"
                            >
                                ✕
                            </button>
                        </div>

                        {/* Mutations table */}
                        <div style={{ marginBottom: '20px' }}>
                            <PredictedMutantTable
                                yamlConfig={yamlConfig}
                                predictedMutantCsvData={evolutionCsvData}
                                setSelectedSubsequence={setSelectedSubsequence}
                            />
                        </div>

                        {/* Render plotly charts with the debug data */}
                        <DebugPlots debugData={evolutionDebugData} />
                    </TableSection>
                    : null
            }

            {/* Collapsible New Run Section */}
            <CollapsibleSection
                title="New Evolution Run"
                isOpen={showForm}
                onToggle={() => setShowForm(!showForm)}
            >
                {/* Help Alert */}
                <Alert
                    message="What is an Evolution Run?"
                    description={
                        <div>
                            <Paragraph>
                                In an Evolution run, a machine learning model is trained on your protein activity measurements, and that model is used to predict the activity of many other possible mutations. Then a slate of mutants is recommended for screening in the next round.

                                This tool facilitates low-N directed evolution of proteins,
                                with as little as 16 screened mutants per round.
                            </Paragraph>
                            <AntButton
                                type="link"
                                icon={<QuestionCircleOutlined />}
                                onClick={() => setShowHelpModal(true)}
                                style={{ padding: 0 }}
                            >
                                View detailed setup instructions
                            </AntButton>
                        </div>
                    }
                    type="info"
                    showIcon
                    style={{ marginBottom: '20px' }}
                />

                {/* Detailed Help Modal */}
                <Modal
                    title="Evolution Run Setup Guide"
                    open={showHelpModal}
                    onCancel={() => setShowHelpModal(false)}
                    footer={[
                        <AntButton key="close" onClick={() => setShowHelpModal(false)}>
                            Close
                        </AntButton>
                    ]}
                    width={700}
                >
                    <div>
                        <Title level={4}>Required Inputs</Title>
                        <Paragraph>
                            Each evolution run requires:
                        </Paragraph>
                        <ul>
                            <li>
                                <Text strong>Activity Excel File:</Text> A file with columns 'seq_id' and 'activity' containing
                                your measured mutant activities
                            </li>
                            <li>
                                <Text strong>Embedding Files:</Text> Embeddings generated in the Embed tab, containing
                                embeddings for both measured mutants and all mutants you wish to screen
                            </li>
                            <li>
                                <Text strong>Naturalness Files:</Text> Naturalness scores for single mutants of the protein. We recommend using ESM-C 600M
                            </li>
                        </ul>

                        <Title level={4}>Example Activity File Format</Title>
                        <div style={{ textAlign: 'center', margin: '20px 0' }}>
                            <img
                                style={{ width: "300px", border: '1px solid #d9d9d9', borderRadius: '4px' }}
                                src={`/evolve_activity_excel_example.png`}
                                alt="Example activity file format showing seq_id and activity columns"
                            />
                        </div>

                        <Title level={4}>Mode Selection Guide</Title>
                        <ul>
                            <li><Text strong>MLP Few Shot Model:</Text> Recommended for most use cases</li>
                            <li><Text strong>Random Forest Few Shot Model:</Text> Alternative ML approach</li>
                            <li><Text strong>Legacy modes:</Text> Older implementations, use new modes when possible</li>
                        </ul>

                        <Title level={4}>Parameters</Title>
                        <Paragraph>
                            <Text strong>Few Shot Parameters:</Text> JSON configuration for the ML model.
                            Use presets for common configurations or customize with your own JSON.
                        </Paragraph>

                        <Alert
                            message="Estimated Cost"
                            description="~$0.05 per evolution round"
                            type="success"
                            showIcon
                            style={{ marginTop: '16px' }}
                        />

                        <Paragraph style={{ marginTop: '16px' }}>
                            Once complete, you can download the predicted activities for all mutants from the Files tab.
                        </Paragraph>
                    </div>
                </Modal>

                <Card>
                    <Form layout="vertical" style={{ maxWidth: '800px' }}>
                        <Row gutter={16}>
                            <Col span={12}>
                                <Form.Item
                                    label="Evolution Name"
                                    required
                                    help="Give your evolution run a descriptive name"
                                >
                                    <Input
                                        value={evolutionName}
                                        onChange={(e) => setEvolutionName(e.target.value)}
                                        placeholder="e.g., round1_high_activity"
                                    />
                                </Form.Item>
                            </Col>
                            <Col span={12}>
                                <Form.Item
                                    label="Slate Size"
                                    required
                                    help="How many top mutants to recommend"
                                >
                                    <InputNumber
                                        value={numMutants}
                                        onChange={(value) => setNumMutants(value || 24)}
                                        min={1}
                                        style={{ width: '100%' }}
                                    />
                                </Form.Item>
                            </Col>
                        </Row>

                        <Divider>File Selection</Divider>

                        <Form.Item
                            label="Activity File"
                            required
                            help="Excel file with seq_id and activity columns"
                        >
                            <Upload
                                beforeUpload={(file) => {
                                    setActivityFile(file);
                                    return false; // Prevent auto upload
                                }}
                                accept=".xlsx,.xls"
                                maxCount={1}
                                fileList={activityFile ? [{
                                    uid: '1',
                                    name: activityFile.name,
                                    status: 'done'
                                }] : []}
                                onRemove={() => setActivityFile(null)}
                            >
                                <AntButton icon={<UploadOutlined />}>
                                    Select Activity File (.xlsx/.xls)
                                </AntButton>
                            </Upload>
                        </Form.Item>

                        <Form.Item
                            label="Multi-Mutant Embedding Files"
                            required
                            help="Select embedding files generated in the Embed tab. This defines the pool of mutants that will be evaluated."
                        >
                            <Select
                                mode="multiple"
                                value={selectedEmbeddingPaths}
                                onChange={setSelectedEmbeddingPaths}
                                style={{ width: '100%' }}
                                placeholder="Select embedding files"
                            >
                                {availableEmbeddingFiles.map(file => (
                                    <Select.Option key={file.key} value={file.key}>
                                        {file.key.split('/').pop() || file.key}
                                    </Select.Option>
                                ))}
                            </Select>
                        </Form.Item>

                        <Form.Item
                            label="Single Mutant Naturalness Files"
                            help="Select naturalness files. We recommend using ESM-C 600M."
                        >
                            <Select
                                mode="multiple"
                                value={selectedNaturalnessPaths}
                                onChange={setSelectedNaturalnessPaths}
                                style={{ width: '100%' }}
                                placeholder="Select naturalness files"
                            >
                                {availableNaturalnessFiles.map(file => (
                                    <Select.Option key={file.key} value={file.key}>
                                        {file.key.split('/').pop() || file.key}
                                    </Select.Option>
                                ))}
                            </Select>
                        </Form.Item>

                        <Divider>Model Parameters</Divider>

                        <Form.Item
                            label="Presets"
                            help="Choose a preset configuration or select 'Custom' to define your own"
                        >
                            <Select
                                value={selectedPreset}
                                onChange={(preset) => {
                                    setSelectedPreset(preset);
                                    if (preset !== 'custom') {
                                        const presetConfig = FEW_SHOT_PRESETS[preset as keyof typeof FEW_SHOT_PRESETS];
                                        setFewShotParams(presetConfig.params);
                                        if (presetConfig.mode) {
                                            setMode(presetConfig.mode);
                                        }
                                    }
                                }}
                                style={{ width: '100%' }}
                            >
                                <Select.Option value="folde_default_mlp">FolDE Default MLP</Select.Option>
                                <Select.Option value="evolvepro">EvolvePro</Select.Option>
                                <Select.Option value="custom">Custom</Select.Option>
                            </Select>
                        </Form.Item>

                        <Form.Item
                            label="Model Choice"
                            required
                            help="ML algorithm to use for predictions"
                        >
                            <Select
                                value={mode}
                                onChange={(newMode) => {
                                    setMode(newMode);
                                    // If user manually changes model choice, switch to custom preset
                                    if (selectedPreset !== 'custom') {
                                        setSelectedPreset('custom');
                                    }
                                }}
                                style={{ width: '100%' }}
                            >
                                <Select.Option value="TorchMLPFewShotModel">MLP Few Shot Model (Recommended)</Select.Option>
                                <Select.Option value="RandomForestFewShotModel">Random Forest Few Shot Model</Select.Option>
                                <Select.Option value="randomforest">(Legacy) Random Forest</Select.Option>
                                <Select.Option value="mlp">(Legacy) Multi-Layer Perceptron</Select.Option>
                                <Select.Option value="finetuning">(Legacy) Finetuning</Select.Option>
                            </Select>
                        </Form.Item>

                        {mode === 'finetuning' && (
                            <Form.Item
                                label="Model Checkpoint"
                                help="Pre-trained model to fine-tune (legacy mode only)"
                            >
                                <Select
                                    value={finetuningModelCheckpoint}
                                    onChange={setFinetuningModelCheckpoint}
                                    style={{ width: '100%' }}
                                >
                                    <Select.Option value="facebook/esm2_t6_8M_UR50D">ESM2 (8M params)</Select.Option>
                                    <Select.Option value="facebook/esm2_t33_650M_UR50D">ESM2 (650M params)</Select.Option>
                                    <Select.Option value="facebook/esm2_t48_15B_UR50D">ESM2 (15B params)</Select.Option>
                                </Select>
                            </Form.Item>
                        )}

                        <Form.Item
                            label="Few Shot Parameters (JSON format)"
                            help="Model-specific parameters in JSON format. Border color indicates validity."
                            validateStatus={jsonValidationStatus}
                        >
                            <Input.TextArea
                                value={fewShotParams}
                                onChange={(e) => {
                                    const value = e.target.value;
                                    setFewShotParams(value);
                                    // If user manually edits, switch to custom
                                    if (selectedPreset !== 'custom') {
                                        const presetConfig = FEW_SHOT_PRESETS[selectedPreset as keyof typeof FEW_SHOT_PRESETS];
                                        if (value !== presetConfig.params) {
                                            setSelectedPreset('custom');
                                        }
                                    }
                                }}
                                placeholder='{"key": "value"}'
                                rows={4}
                                style={{
                                    fontFamily: 'monospace',
                                    borderColor: jsonValidationStatus === 'success' ? '#52c41a' :
                                        jsonValidationStatus === 'error' ? '#ff4d4f' : undefined,
                                    borderWidth: jsonValidationStatus ? '2px' : undefined
                                }}
                            />
                        </Form.Item>

                        <Form.Item>
                            <AntButton
                                type="primary"
                                icon={<PlayCircleOutlined />}
                                onClick={handleEvolve}
                                disabled={
                                    evolutionName === '' ||
                                    !activityFile ||
                                    ((mode === 'randomforest' || mode === 'mlp') && selectedEmbeddingPaths.length === 0) ||
                                    (mode === 'finetuning' && !finetuningModelCheckpoint)
                                }
                                size="large"
                            >
                                Start Evolution
                            </AntButton>
                        </Form.Item>
                    </Form>
                </Card>
            </CollapsibleSection>
        </TabContainer>
    );
};

export default EvolveTab;
