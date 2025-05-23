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



type RowData = {
    seqId: string;
    footprint: string;
    relevantMeasuredMutants: string;
    predictionMean: number;
    predictionStddev: number;
    score: number;
    modelPredictions?: number[];
}

const seqIdToFootprint = (seqId: string): string => {
    // Convert seqIds like A3C_G56Y_Y79T into fooprints like 3_56_79

    // First split by underscore.
    const alleleIds = seqId.split('_');
    const loci = alleleIds.map((alleleId) => {
        // alleleId[1:-1];
        return alleleId.slice(1, -1);
    });
    return loci.join('_');
}

const parseCsvDataIntoRowData = (predictedMutantCsvDataString: string, beta: number, maxPerFootprint: number, topPerformersToDisplay: number | undefined): RowData[] | null => {
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

    const interiorTableRows = data.map((row) => {
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
        const score = mean + beta * stddev;

        const footprint = seqIdToFootprint(row['seq_id']);

        return {
            seqId: row['seq_id'],
            relevantMeasuredMutants: row['relevant_measured_mutants'],
            footprint: footprint,
            predictionMean: mean,
            predictionStddev: stddev,
            score: score,
            modelPredictions: predictions,
        };
    });

    const allRows = interiorTableRows.sort((a, b) => b.score - a.score);

    // Filter out mutations where we've already seen that locus N times.
    const footprintCounts: { [key: string]: number } = {};
    const relevantRows = [];
    for (const row of allRows) {
        footprintCounts[row.footprint] = (footprintCounts[row.footprint] || 0) + 1;

        if (maxPerFootprint && footprintCounts[row.footprint] > maxPerFootprint) {
            continue;
        }
        relevantRows.push(row);
        if (topPerformersToDisplay && relevantRows.length >= topPerformersToDisplay) {
            break;
        }
    }
    return relevantRows;
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


// Add a utility function to calculate Pearson correlation
function calculateCorrelation(x: number[], y: number[]): number {
    const n = x.length;
    if (n === 0 || n !== y.length) return 0;

    // Calculate means
    const xMean = x.reduce((sum, val) => sum + val, 0) / n;
    const yMean = y.reduce((sum, val) => sum + val, 0) / n;

    // Calculate correlation
    let numerator = 0;
    let xDenominator = 0;
    let yDenominator = 0;

    for (let i = 0; i < n; i++) {
        const xDiff = x[i] - xMean;
        const yDiff = y[i] - yMean;
        numerator += xDiff * yDiff;
        xDenominator += xDiff * xDiff;
        yDenominator += yDiff * yDiff;
    }

    const denominator = Math.sqrt(xDenominator * yDenominator);
    return denominator === 0 ? 0 : numerator / denominator;
}


interface PredictedMutantTableProps {
    yamlConfig: string | null;
    predictedMutantCsvData: string | null;
    beta: number;
    maxPerFootprint: number;
    topPerformersToDisplay: number;
    setSelectedSubsequence: (selection: Selection | null) => void;
}


// Now modify the PredictedMutantTable component to include the heatmap
const PredictedMutantTable: React.FC<PredictedMutantTableProps> = ({
    yamlConfig,
    predictedMutantCsvData,
    beta,
    maxPerFootprint,
    topPerformersToDisplay,
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
        const data = parseCsvDataIntoRowData(predictedMutantCsvData, beta, maxPerFootprint, topPerformersToDisplay);
        if (!data) return null;

        if (data && sortColumn) {
            return [...data].sort((a, b) => {
                const aValue = a[sortColumn as keyof RowData];
                const bValue = b[sortColumn as keyof RowData];
                if (aValue === null) return 1;
                if (bValue === null) return -1;
                return sortDirection === 'ASC'
                    ? (aValue < bValue ? -1 : 1)
                    : (aValue > bValue ? -1 : 1);
            });
        }
        return data;
    }, [predictedMutantCsvData, beta, maxPerFootprint, topPerformersToDisplay, sortColumn, sortDirection]);

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
            key: "seqId",
            name: "Sequence ID",
            sortable: true,
            resizable: true,
            sortDescendingFirst: true,
            formatter: ({ row }: { row: any }) => (
                <div uk-tooltip={row.seqId}>{row.seqId}</div>
            )
        },
        {
            key: 'relevantMeasuredMutants',
            name: "Measured",
            resizable: true,
            maxWidth: 200,
            formatter: ({ row }: { row: any }) => (
                <div uk-tooltip={row.relevantMeasuredMutants}>{row.relevantMeasuredMutants}</div>
            )
        },
        {
            key: 'predictionMean',
            name: "Mean",
            resizable: true,
            formatter: ({ row }: { row: any }) => row.predictionMean.toFixed(2),
        },
        {
            key: "predictionStddev",
            name: "STD",
            resizable: true,
            formatter: ({ row }: { row: any }) => row.predictionStddev.toFixed(2),
        },
        {
            key: "score",
            name: "Score",
            sortable: true,
            resizable: true,
            formatter: ({ row }: { row: any }) => row.score.toFixed(2),
            sortDescendingFirst: true
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
                onRowClick={(e, row) => {
                    setSelectedSeqIds([row.seqId]);
                }}
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
                                text: correlationData.map(row =>
                                    row.map(val => val.toFixed(2))
                                ),
                                hovertemplate: '%{x} vs %{y}<br>Correlation: %{text}<extra></extra>',
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
        if (model.train_loss && model.train_loss.some(val => val !== 0 && val !== null)) {
            // Filter out zeros which appear to be placeholders
            const nonZeroTrainLoss = model.train_loss.map(val => val === 0 ? null : val);

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

        if (model.val_loss && model.val_loss.some(val => val !== 0 && val !== null)) {
            // Filter out zeros which appear to be placeholders
            const nonZeroValLoss = model.val_loss.map(val => val === 0 ? null : val);

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
    const [selectedEmbeddingPaths, setSelectedEmbeddingPaths] = useState<string[]>([]);
    const [selectedNaturalnessPaths, setSelectedNaturalnessPaths] = useState<string[]>([]);
    const [finetuningModelCheckpoint, setFinetuningModelCheckpoint] = useState<string>('facebook/esm2_t6_8M_UR50D');
    const [fewShotParams, setFewShotParams] = useState<string>('');

    const [displayedEvolutionId, setDisplayedEvolutionId] = useState<number | null>(null);
    const [evolutionCsvData, setEvolutionCsvData] = useState<string | null>(null);
    const [evolutionDebugData, setEvolutionDebugData] = useState<any>(null);
    const [beta, setBeta] = useState<number>(1.0);
    const [maxMutationsPerFootprint, setMaxMutationsPerFootprint] = useState<number>(2);
    const [topPerformersToDisplay, setTopPerformersToDisplay] = useState<number>(24);

    const availableEmbeddingFiles = files?.filter(file =>
        file.key.includes('embed')
    ) || [];
    const availableNaturalnessFiles = files?.filter(file =>
        file.key.includes('naturalness')
    ) || [];

    const handleEmbeddingFileSelection = (event: ChangeEvent<HTMLSelectElement>) => {
        const selectedOptions = Array.from(event.target.selectedOptions).map(option => option.value);
        setSelectedEmbeddingPaths(selectedOptions);
    };

    const handleNaturalnessFileSelection = (event: ChangeEvent<HTMLSelectElement>) => {
        const selectedOptions = Array.from(event.target.selectedOptions).map(option => option.value);
        setSelectedNaturalnessPaths(selectedOptions);
    };

    const handleActivityFileUpload = (event: ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            if (!file.name.match(/\.(xlsx|xls)$/i)) {
                notify.error('Please upload an Excel file (.xlsx or .xls)');
                return;
            }
            setActivityFile(file);
        }
    };

    const handleEvolve = async () => {
        if (!activityFile || (selectedEmbeddingPaths.length === 0)) {
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
        getFile(evolution.fold_id, predictedActivityPath).then(
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
            (e) => {
                notify.success(`Evolution ${evolutionId} deleted.`);
            },
            (e) => {
                notify.error(e.toString());
            }
        )
    }



    return (
        <TabContainer>
            {/* Description Section */}
            <DescriptionSection title="Evolution Runs Overview">
                This section allows you to run a version of
                <a href="https://www.biorxiv.org/content/10.1101/2024.07.17.604015v1"> EvolvePro </a>
                on your protein. This tool facilitates low-N directed evolution of proteins,
                with as little as 16 screened mutants per round. Please see the paper for more
                details.
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

                        <NumberInputControl
                            label="Beta (Upper Confidence Bound parameter)"
                            value={beta}
                            onChange={setBeta}
                            min={0}
                            step={0.1}
                        />
                        <NumberInputControl
                            label="Max number of mutants per footprint"
                            value={maxMutationsPerFootprint}
                            onChange={setMaxMutationsPerFootprint}
                            min={1}
                        />
                        <NumberInputControl
                            label="Top mutants to display"
                            value={topPerformersToDisplay}
                            onChange={setTopPerformersToDisplay}
                            min={1}
                        />
                        {/* Mutations table */}
                        <div style={{ marginBottom: '20px' }}>
                            <h3 style={{ marginBottom: '15px', color: '#444' }}>Selected Mutants</h3>
                            <PredictedMutantTable
                                yamlConfig={yamlConfig}
                                predictedMutantCsvData={evolutionCsvData}
                                beta={beta}
                                maxPerFootprint={maxMutationsPerFootprint}
                                topPerformersToDisplay={topPerformersToDisplay}
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
                <h3>Start New Evolution Run</h3>
                Each run takes in an
                <ul>
                    <li><code>activity excel file</code> with columns seq_id and activity</li>
                    <li><code>embedding files</code> embeddings run in the excel tab, containing embeddings for both the mutants with activity measurements as well as all mutants you wish to screen.</li>
                </ul>
                <p>
                    Once complete, you can download the predicted activities for all mutants from the Files tab.
                </p>
                <h4>
                    Example activity file
                </h4>
                <img
                    style={{
                        width: "200px",
                    }}
                    src={`/evolve_activity_excel_example.png`}
                    alt=""
                />
                <p>
                    <code>Estimated cost:</code>~$0.05 per evolution round.
                </p>
                <FormRow>
                    <FormField>
                        <TextInputControl
                            label="Name"
                            value={evolutionName}
                            onChange={setEvolutionName}
                        />
                    </FormField>

                    <FormField>
                        <FileUploadControl
                            label="Upload Activity File"
                            onChange={setActivityFile}
                            accept=".xlsx,.xls"
                            selectedFile={activityFile}
                        />
                    </FormField>

                    <FormField>
                        <SelectControl
                            label="Mode"
                            value={mode}
                            onChange={setMode}
                            options={[
                                { value: "TorchMLPFewShotModel", label: "MLP Few Shot Model" },
                                { value: "RandomForestFewShotModel", label: "RandomForestFewShotModel" },
                                { value: "randomforest", label: "(old) Random Forest" },
                                { value: "mlp", label: "(old) Multi-Layer Perceptron" },
                                { value: "finetuning", label: "(old) Finetuning" }
                            ]}
                        />
                    </FormField>

                    {/* Conditional inputs based on mode */}
                    {mode === 'finetuning' && (
                        <FormField>
                            <SelectControl
                                label="Model Checkpoint"
                                value={finetuningModelCheckpoint}
                                onChange={setFinetuningModelCheckpoint}
                                options={[
                                    { value: "facebook/esm2_t6_8M_UR50D", label: "ESM2 (8M params)" },
                                    { value: "facebook/esm2_t33_650M_UR50D", label: "ESM2 (650M params)" },
                                    { value: "facebook/esm2_t48_15B_UR50D", label: "ESM2 (15B params)" }
                                ]}
                            />
                        </FormField>
                    )}
                </FormRow>

                <MultiSelectControl
                    label="Select Embedding Files"
                    options={availableEmbeddingFiles.map(file => ({
                        key: file.key,
                        label: file.key.split('/').pop() || file.key
                    }))}
                    selectedValues={selectedEmbeddingPaths}
                    onChange={setSelectedEmbeddingPaths}
                    style={{ width: '100%' }}
                />

                <MultiSelectControl
                    label="Select Naturalness Files"
                    options={availableNaturalnessFiles.map(file => ({
                        key: file.key,
                        label: file.key.split('/').pop() || file.key
                    }))}
                    selectedValues={selectedNaturalnessPaths}
                    onChange={setSelectedNaturalnessPaths}
                    style={{ width: '100%' }}
                />

                <TextAreaControl
                    label="Few Shot Parameters (JSON format)"
                    value={fewShotParams}
                    onChange={(value) => {
                        setFewShotParams(value);
                    }}
                    placeholder='{"key": "value"}'
                    rows={4}
                    inputStyle={{ fontFamily: 'monospace' }}
                    style={{ width: '100%' }}
                />
                <p className="uk-text-meta">
                    Enter a valid JSON object. Border will turn green when valid, red when invalid.
                </p>

                <button
                    className="uk-button uk-button-primary uk-margin-top"
                    onClick={handleEvolve}
                    disabled={
                        evolutionName === '' ||
                        !activityFile ||
                        ((mode === 'randomforest' || mode === 'mlp') && selectedEmbeddingPaths.length === 0) ||
                        (mode === 'finetuning' && !finetuningModelCheckpoint)
                    }
                >
                    Start Evolution
                </button>
            </CollapsibleSection>
        </TabContainer>
    );
};

export default EvolveTab;
