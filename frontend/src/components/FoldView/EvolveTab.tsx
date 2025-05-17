import React, { useState, ChangeEvent, useMemo, useEffect } from 'react';
import UIkit from 'uikit';
import { FileInfo, Evolution, Invokation } from 'src/types/types';
import { evolve } from '../../api/evolveApi';
import { FaDownload, FaEye, FaFileCode, FaRedo } from 'react-icons/fa';
import fileDownload from 'js-file-download';
import { removeLeadingSlash } from '../../api/commonApi';
import { getFile } from '../../api/fileApi';
import { notify } from '../../services/NotificationService';
import Papa from 'papaparse';
import ReactDataGrid from 'react-data-grid';
import { BoltzYamlHelper } from '../../util/boltzYamlHelper';
import { Selection } from './StructurePane';



type RowData = {
    seqId: string;
    footprint: string;
    relevantMeasuredMutants: string;
    predictionMean: number;
    predictionStddev: number;
    score: number;
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


interface PredictedMutantTableProps {
    yamlConfig: string | null;
    predictedMutantCsvData: string | null;
    beta: number;
    maxPerFootprint: number;
    topPerformersToDisplay: number;
    setSelectedSubsequence: (selection: Selection | null) => void;
}

const PredictedMutantTable: React.FC<PredictedMutantTableProps> = ({
    yamlConfig,
    predictedMutantCsvData,
    beta,
    maxPerFootprint,
    topPerformersToDisplay,
    setSelectedSubsequence,
}) => {
    if (!predictedMutantCsvData) return null;

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

    if (!tableData) return null;
    const columns = [
        {
            key: "seqId",
            name: "Sequence ID",
            sortable: true,
            resizable: true,
            sortDescendingFirst: true
        },
        {
            key: 'relevantMeasuredMutants',
            name: "Measured",
            resizable: true,
            maxWidth: 200,
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
        <div style={{ width: "auto", height: "auto", marginTop: "20px" }}>
            <ReactDataGrid
                columns={columns}
                rowGetter={i => tableData[i]}
                rowsCount={tableData.length}
                // enableRowSelect={true}
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
            <div style={{ marginTop: '10px', display: 'flex', gap: '10px' }}>
                <button
                    className="uk-button uk-button-default"
                    onClick={() => copyMutationsToClipboard()}
                >
                    Copy mutations to clipboard
                </button>
                <button className="uk-button uk-button-primary" onClick={() => highlightResiduesOnModel()}>
                    Highlight residues on model
                </button>
            </div>
        </div>
    );
};

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

        getFile(foldId, `evolve/${evolution.name}/predicted_activity.csv`).then(
            (fileBlob: Blob) => {
                // Create a FileReader to read the blob as text
                const reader = new FileReader();
                reader.onload = (e) => {
                    const fileString = e.target?.result as string;
                    setEvolutionCsvData(fileString);
                };
                reader.readAsText(fileBlob);
            },
            (e) => {
                console.log(e);
                notify.error(e.toString());
            }
        );
    }



    return (
        <div style={{ padding: '20px', backgroundColor: '#f8f9fa', boxShadow: '0 2px 6px rgba(0, 0, 0, 0.1)', borderRadius: '8px' }}>
            {/* Description Section */}
            <section style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#ffffff', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                <h3 style={{ marginBottom: '10px' }}>Evolution Runs Overview</h3>
                <div>
                    This section allows you to run a version of
                    <a href="https://www.biorxiv.org/content/10.1101/2024.07.17.604015v1"> EvolvePro </a>
                    on your protein. This tool facilitates low-N directed evolution of proteins,
                    with as little as 16 screened mutants per round. Please see the paper for more
                    details. Each run takes in an
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
                </div>
            </section>

            {/* Evolution Runs Table */}
            <section style={{ marginBottom: '30px', padding: '15px', backgroundColor: '#ffffff', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)', overflowX: 'scroll' }}>
                <h3>Evolution Runs</h3>
                <table className="uk-table uk-table-striped">
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
                                <td>{evolution.name}</td>
                                <td>{getEvolutionStatus(evolution)}</td>
                                <td>
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
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </section>


            {
                displayedEvolutionId ?
                    <>
                        <div>
                            <label>
                                Beta (Upper Confidence Bound parameter):
                                <input
                                    type="number"
                                    className="uk-input"
                                    value={beta}
                                    onChange={(e) => setBeta(parseFloat(e.target.value))}
                                    style={{ width: '100px', marginLeft: '10px' }}
                                    min="0"
                                />
                            </label>
                        </div>
                        <div>
                            <label>
                                Max number of mutants per footprint:
                                <input
                                    type="number"
                                    className="uk-input"
                                    value={maxMutationsPerFootprint}
                                    onChange={(e) => setMaxMutationsPerFootprint(parseInt(e.target.value))}
                                    style={{ width: '100px', marginLeft: '10px' }}
                                    min="1"
                                />
                            </label>
                        </div>
                        <div>
                            <label>
                                Top mutants to display:
                                <input
                                    type="number"
                                    className="uk-input"
                                    value={topPerformersToDisplay}
                                    onChange={(e) => setTopPerformersToDisplay(parseInt(e.target.value))}
                                    style={{ width: '100px', marginLeft: '10px' }}
                                    min="1"
                                />
                            </label>
                        </div>
                        <PredictedMutantTable
                            yamlConfig={yamlConfig}
                            predictedMutantCsvData={evolutionCsvData}
                            beta={beta}
                            maxPerFootprint={maxMutationsPerFootprint}
                            topPerformersToDisplay={topPerformersToDisplay}
                            setSelectedSubsequence={setSelectedSubsequence}
                        />
                    </>
                    : null
            }

            {/* Collapsible New Run Section */}
            <div>
                <div
                    className='uk-margin-top uk-margin-bottom'
                    style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        padding: "10px 15px",
                        backgroundColor: "#f8f9fa",
                        border: "1px solid #e0e0e0",
                        borderRadius: "8px",
                        boxShadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
                        cursor: "pointer",
                        fontWeight: "bold",
                    }}
                    onClick={() => setShowForm(!showForm)}
                >
                    <span>New Evolution Run</span>
                    <span>{showForm ? "▲" : "▼"}</span>
                </div>
                {showForm && (
                    <section style={{ padding: '15px', backgroundColor: '#ffffff', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                        <h3>Start New Evolution Run</h3>
                        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>

                            {/* Name Input */}
                            <div style={{ flex: 1, minWidth: '200px' }}>
                                <label className="uk-form-label">Name</label>
                                <input
                                    type="text"
                                    className="uk-input"
                                    value={evolutionName}
                                    onChange={(e) => setEvolutionName(e.target.value)}
                                />
                            </div>

                            {/* Activity File Upload */}
                            <div style={{ flex: 1, minWidth: '200px' }}>
                                <label className="uk-form-label">Upload Activity File</label>
                                <input
                                    type="file"
                                    accept=".xlsx,.xls"
                                    onChange={handleActivityFileUpload}
                                    className="uk-input"
                                />
                                {activityFile && (
                                    <p className="uk-text-meta">Selected file: {activityFile.name}</p>
                                )}
                            </div>

                            {/* Mode Selection */}
                            <div style={{ flex: 1, minWidth: '200px' }}>
                                <label className="uk-form-label">Mode</label>
                                <select
                                    className="uk-select"
                                    value={mode}
                                    onChange={(e) => setMode(e.target.value)}
                                >
                                    <option value="TorchMLPFewShotModel">MLP Few Shot Model</option>
                                    <option value="RandomForestFewShotModel">RandomForestFewShotModel</option>
                                    <option value="randomforest">(old) Random Forest</option>
                                    <option value="mlp">(old) Multi-Layer Perceptron</option>
                                    <option value="finetuning">(old) Finetuning</option>
                                </select>
                            </div>

                            {/* Conditional inputs based on mode */}
                            {mode === 'finetuning' && (
                                <div style={{ flex: 1, minWidth: '200px' }}>
                                    <label className="uk-form-label">Model Checkpoint</label>
                                    <select
                                        className="uk-select"
                                        value={finetuningModelCheckpoint}
                                        onChange={(e) => setFinetuningModelCheckpoint(e.target.value)}
                                    >
                                        <option value="facebook/esm2_t6_8M_UR50D">ESM2 (8M params)</option>
                                        <option value="facebook/esm2_t33_650M_UR50D">ESM2 (650M params)</option>
                                        <option value="facebook/esm2_t48_15B_UR50D">ESM2 (15B params)</option>
                                    </select>
                                </div>
                            )}

                            <div style={{ flex: '0 0 auto', width: '100%' }}>
                                <label className="uk-form-label">Select Embedding Files</label>
                                <select
                                    className="uk-select"
                                    multiple
                                    size={Math.min(10, availableEmbeddingFiles.length || 1)}
                                    value={selectedEmbeddingPaths}
                                    onChange={handleEmbeddingFileSelection}
                                >
                                    {availableEmbeddingFiles.map(file => (
                                        <option key={file.key} value={file.key}>
                                            {file.key.split('/').pop()}
                                        </option>
                                    ))}
                                </select>
                                <p className="uk-text-meta">
                                    Selected {selectedEmbeddingPaths.length} embedding file(s)
                                </p>
                            </div>

                            <div style={{ flex: '0 0 auto', width: '100%' }}>
                                <label className="uk-form-label">Select Naturalness Files</label>
                                <select
                                    className="uk-select"
                                    multiple
                                    size={Math.min(10, availableNaturalnessFiles.length || 1)}
                                    value={selectedNaturalnessPaths}
                                    onChange={handleNaturalnessFileSelection}
                                >
                                    {availableNaturalnessFiles.map(file => (
                                        <option key={file.key} value={file.key}>
                                            {file.key.split('/').pop()}
                                        </option>
                                    ))}
                                </select>
                                <p className="uk-text-meta">
                                    Selected {selectedNaturalnessPaths.length} naturalness file(s)
                                </p>
                            </div>

                            {/* New Few Shot Parameters Input */}
                            <div style={{ flex: '0 0 auto', width: '100%' }}>
                                <label className="uk-form-label">Few Shot Parameters (JSON format)</label>
                                <textarea
                                    className="uk-textarea"
                                    rows={4}
                                    value={fewShotParams}
                                    onChange={(e) => {
                                        setFewShotParams(e.target.value);
                                        // Try to validate JSON
                                        try {
                                            if (e.target.value) {
                                                JSON.parse(e.target.value);
                                                e.target.style.borderColor = '#32d296'; // Success color
                                            } else {
                                                e.target.style.borderColor = ''; // Default color
                                            }
                                        } catch (err) {
                                            e.target.style.borderColor = '#f0506e'; // Error color
                                        }
                                    }}
                                    placeholder='{"key": "value"}'
                                    style={{ fontFamily: 'monospace' }}
                                />
                                <p className="uk-text-meta">
                                    Enter a valid JSON object. Border will turn green when valid, red when invalid.
                                </p>
                            </div>
                        </div>

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
                    </section>
                )}
            </div>
        </div >
    );
};

export default EvolveTab;
