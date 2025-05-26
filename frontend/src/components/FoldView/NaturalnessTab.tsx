import React, { useState, useMemo } from 'react';
import { Logit, Invokation } from 'src/types/types';
import { FaDownload, FaEye, FaFileCode, FaRedo } from 'react-icons/fa';
import { downloadFileStraightToFilesystem, getFile } from '../../api/fileApi';
import { startLogits } from '../../api/embedApi';
import Plot from 'react-plotly.js';
import { Data } from 'plotly.js';
import Papa from 'papaparse';
import { ESMModelPicker } from './ESMModelPicker';
import { Selection } from './StructurePane';
import ReactDataGrid from 'react-data-grid';
import { notify } from '../../services/NotificationService';
import { BoltzYamlHelper } from '../../util/boltzYamlHelper';
import { TabContainer, DescriptionSection, TableSection, CollapsibleSection, FormRow, FormField, ButtonGroup, ResponsiveTable } from '../../util/tabComponents';
import { TextInputControl, CheckboxControl, NumberInputControl } from '../../util/controlComponents';
import { DataTableContainer } from '../../util/plotComponents';
import { Alert, Modal, Button as AntButton, Typography } from 'antd';
import { QuestionCircleOutlined } from '@ant-design/icons';

const { Text, Paragraph, Title } = Typography;


const NATURALNESS_COLUMN = 'probability';
const WT_MARGINAL_COLUMN = 'wt_marginal';

// Define standard amino acid residues
const RESIDUES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'];

interface NaturalnessTabProps {
    foldId: number;
    foldName: string | null;
    yamlConfig: string | null;
    jobs: Invokation[] | null;
    logits: Logit[] | null;
    setSelectedSubsequence: (selection: Selection | null) => void;
    openUpLogsForJob: (jobId: number | undefined) => void;
}



type RowData = {
    seqId: string;
    score: number;
    model: number | null;
}

const parseSeqId = (seqId: string): { wtResidue: string, locus: number, mutantResidue: string } => {
    // If there is an underscore in seq id, we bail.
    if (seqId.includes('_')) {
        throw new Error(`Invalid seqId: "${seqId}"`);
    }
    const match = seqId.match(/([A-Z])(\d+)([A-Z])/);
    if (!match) {
        throw new Error(`Invalid seqId: "${seqId}"`);
    }
    return { wtResidue: match[1], locus: parseInt(match[2]), mutantResidue: match[3] };
}

const parseCsvDataIntoRowData = (logitCsvDataString: string, useWtMarginalAsScore: boolean, zeroWildType: boolean, maxMutationsPerLocus: number | undefined, topPerformersToDisplay: number | undefined): RowData[] | null => {
    const { data, errors } = Papa.parse<Record<string, string>>(logitCsvDataString, {
        header: true,
        delimiter: ',',
        skipEmptyLines: true,
        dynamicTyping: true
    });

    if (errors.length > 0) {
        notify.error(`Error parsing logit CSV: ${errors.map(error => error.message).join(', ')}`);
        return null;
    }

    const interiorTableRows = data.filter((row) => {
        // Filter out rows that end in special characters like <cls>.
        const endsInSpecialCharacter = row['seq_id'].match(/.*<.*>/);
        const endsInDot = row['seq_id'].match(/.*\..*/);
        const endsInHyphen = row['seq_id'].match(/.*-.*/);
        const endsInBar = row['seq_id'].match(/.*\|.*/);
        if (endsInSpecialCharacter || endsInDot || endsInHyphen || endsInBar) {
            console.log(`Filtering out row: ${row['seq_id']}`);
            return false;
        }
        return true;
    }).map((row) => {
        var score;
        if (useWtMarginalAsScore) {
            score = parseFloat(row[WT_MARGINAL_COLUMN]);
            // Take the log of the score.
            score = score ? Math.log(score || 1e-7) : null;
        } else {
            score = parseFloat(row[NATURALNESS_COLUMN]) || 0;
        }
        score = score || 0;

        const { wtResidue, locus, mutantResidue } = parseSeqId(row['seq_id']);

        if (zeroWildType && wtResidue == mutantResidue) {
            score = 0;
        }
        var model = null;
        if (row['model'] != null) {
            model = parseFloat(row['model']);
        }

        return {
            seqId: row['seq_id'],
            score: score,
            model: model,
        };
    });

    const allRows = interiorTableRows.sort((a, b) => b.score - a.score);

    // Filter out mutations where we've already seen that locus N times.
    const locusCounts: { [key: number]: number } = {};
    const relevantRows = [];
    for (const row of allRows) {
        const { wtResidue, locus, mutantResidue } = parseSeqId(row.seqId);

        locusCounts[locus] = (locusCounts[locus] || 0) + 1;

        if (maxMutationsPerLocus && locusCounts[locus] > maxMutationsPerLocus) {
            continue;
        }
        relevantRows.push(row);
        if (topPerformersToDisplay && relevantRows.length >= topPerformersToDisplay) {
            break;
        }
    }
    return relevantRows;
}

// const LogitTable: React.FC<LogitTableProps> = ({ logitCsvData, useWtMarginalAsScore, zeroWildType, topPerformersToDisplay }) => {
//     if (!logitCsvData) return null;

//     const tableData: RowData[] | null = useMemo(() => {
//         return parseCsvDataIntoRowData(logitCsvData, useWtMarginalAsScore, zeroWildType)?.slice(0, topPerformersToDisplay) || null;
//     }, [logitCsvData, useWtMarginalAsScore, topPerformersToDisplay]);
//     if (!tableData) return null;

//     const tableSubset = tableData.slice(0, topPerformersToDisplay);

//     let columnDefs: ColDef<RowData>[] = [
//         { field: 'seqId', headerName: 'Sequence ID', sortable: true, filter: true },
//         {
//             field: 'score',
//             headerName: useWtMarginalAsScore ? 'WT Marginal Likelihood' : 'Probability',
//             sortable: true,
//             filter: 'agNumberColumnFilter',
//             valueFormatter: (params: any) => params.value.toExponential(6),
//             sort: 'desc',
//             sortIndex: 0
//         },
//         { field: 'model', headerName: 'Model', sortable: true, filter: true }
//     ];

//     return (
//         <div
//             className="ag-theme-alpine"
//             style={{
//                 width: '100%',
//                 height: '500px',
//                 marginTop: '20px'
//             }}
//         >
//             <AgGridReact
//                 modules={[ClientSideRowModelModule]}
//                 rowData={tableSubset}
//                 columnDefs={columnDefs}
//                 defaultColDef={{
//                     flex: 1,
//                     minWidth: 100,
//                     resizable: true,
//                     sortable: true,
//                     filter: true,
//                     suppressMovable: true,
//                     // cellStyle: { userSelect: 'text' }
//                 }}
//                 enableCellTextSelection={true}
//                 copyHeadersToClipboard={true}
//                 domLayout='autoHeight'
//                 ensureDomOrder={true}
//             // suppressCellFocus={true}
//             // suppressRowClickSelection={true}
//             />
//         </div>
//     );
// };
interface LogitTableProps {
    logitCsvData: string | null;
    useWtMarginalAsScore: boolean;
    zeroWildType: boolean;
    maxMutationsPerLocus: number;
    topPerformersToDisplay: number;
}

const LogitTable: React.FC<LogitTableProps> = ({
    logitCsvData,
    useWtMarginalAsScore,
    zeroWildType,
    maxMutationsPerLocus,
    topPerformersToDisplay,
}) => {
    if (!logitCsvData) return null;

    const [sortColumn, setSortColumn] = useState<string | null>(null);
    const [sortDirection, setSortDirection] = useState<'ASC' | 'DESC'>('DESC');

    const tableData: RowData[] | null = useMemo(() => {
        const data = parseCsvDataIntoRowData(logitCsvData, useWtMarginalAsScore, zeroWildType, maxMutationsPerLocus, topPerformersToDisplay);
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
    }, [logitCsvData, useWtMarginalAsScore, zeroWildType, maxMutationsPerLocus, topPerformersToDisplay, sortColumn, sortDirection]);

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
            key: "score",
            name: useWtMarginalAsScore ? "Log(WT Marginal Likelihood)" : "Probability",
            sortable: true,
            resizable: true,
            formatter: ({ row }: { row: any }) => row.score.toFixed(4),
            sortDescendingFirst: true
        }
    ];

    return (
        <DataTableContainer>
            <ReactDataGrid
                columns={columns}
                rowGetter={i => tableData[i]}
                rowsCount={tableData.length}
                enableCellSelect={true}
                onGridSort={(sortCol, direction) => {
                    setSortColumn(sortCol);
                    setSortDirection(direction.toUpperCase() as 'ASC' | 'DESC');
                }}
            />
        </DataTableContainer>
    );
};

const NaturalnessTab: React.FC<NaturalnessTabProps> = ({ foldId, foldName, yamlConfig, jobs, logits, setSelectedSubsequence, openUpLogsForJob }) => {
    const [runName, setRunName] = useState<string>('');
    const [logitModel, setLogitModel] = useState<string>('esmc_600m');
    const [useStructure, setUseStructure] = useState<boolean>(false);
    const [getDepthTwoLogits, setGetDepthTwoLogits] = useState<boolean>(false);
    const [showForm, setShowForm] = useState<boolean>(false);

    const [displayedLogitId, setDisplayedLogitId] = useState<number | null>(null);
    const [logitCsvData, setLogitCsvData] = useState<string | null>(null);
    const [maskWildType, setMaskWildType] = useState<boolean>(false);
    const [zeroWildType, setZeroWildType] = useState<boolean>(false);
    const [showWTMarginalLikelihood, setShowWTMarginalLikelihood] = useState<boolean>(true);

    const [maxMutationsPerLocus, setMaxMutationsPerLocus] = useState<number>(2);
    const [topPerformersToDisplay, setTopPerformersToDisplay] = useState<number>(24);


    const handleStartLogit = async () => {
        try {
            notify.info('Starting naturalness run...');
            const logitRun = await startLogits(foldId, runName, logitModel, useStructure, getDepthTwoLogits);
            console.log(`logitRun: ${logitRun}`);
            console.log(`logitRun keys: ${Object.keys(logitRun)}`);
            notify.success(`Logit run started with id ${logitRun.id} and name ${logitRun.name}`);
        } catch (error) {
            notify.error(`Failed to start logit run: ${error}`);
        }
    };

    const getLogitStatus = (logit: Logit): string => {
        const job = jobs?.find(job => job.id === logit.invokation_id);
        return job?.state || 'Unknown';
    };

    const downloadLogitCsv = (logit: Logit) => {
        if (!foldName) {
            notify.warning('Fold name is not set.');
            return;
        }
        const logitPath = `naturalness/logits_${logit.name}_melted.csv`;
        const newFileName = `${foldName}_naturalness_${logit.name}.csv`;
        console.log(`Downloading logits for ${logit.name} at path ${logitPath} to ${newFileName}`);
        downloadFileStraightToFilesystem(
            logit.fold_id,
            logitPath,
            newFileName,
            (progress: number) => {
                console.log(`Downloading ${logitPath}: ${progress}%`);
            }
        );
    };

    const rerunLogit = async (logit: Logit) => {
        notify.info(`Repopulating "New Logit Run" with parameters from ${logit.name}.`);
        setRunName(logit.name);
        setShowForm(true);
        setLogitModel(logit.logit_model);
        setUseStructure(logit.use_structure || false);
        setGetDepthTwoLogits(logit.get_depth_two_logits || false);
    };

    const loadLogit = (logitId: number) => {
        const logit = logits?.find(logit => logit.id === logitId);
        if (!logit) {
            notify.error(`Logit ${logitId} not found.`);
            return;
        }
        setDisplayedLogitId(logitId);
        console.log(`Loading logit ${logit.name}...`);

        getFile(foldId, `naturalness/logits_${logit.name}_melted.csv`).then(
            (fileBlob: Blob) => {
                // Create a FileReader to read the blob as text
                const reader = new FileReader();
                reader.onload = (e) => {
                    const fileString = e.target?.result as string;
                    setLogitCsvData(fileString);
                };
                reader.readAsText(fileBlob);
            },
            (e) => {
                console.log(e);
                notify.error(e.toString());
            }
        );
    }

    const logitPlot = useMemo(() => {
        if (!logitCsvData) return null;

        const rowData = parseCsvDataIntoRowData(logitCsvData, showWTMarginalLikelihood, zeroWildType, undefined, undefined);
        if (!rowData) return null;


        // Process data for heatmap
        const locusSet = new Set<number>();
        const scoreHeatmapData: { [key: string]: number } = {};
        const wtResidues: { [key: number]: string } = {};  // Store wild-type residues by position

        var ensembleMembers = 1.0;

        rowData.forEach(row => {
            if (row.model != null) {
                ensembleMembers = Math.max(ensembleMembers, row.model);
            }
            const seqId = row.seqId;

            // Extract locus and mutant residue using regex
            const match = seqId.match(/([A-Z])(\d+)([A-Z])/);
            if (match) {
                const wtResidue = match[1];
                const locus = parseInt(match[2]);
                const mutantResidue = match[3];
                locusSet.add(locus);
                wtResidues[locus] = wtResidue;
                const key = `${locus}-${mutantResidue}`;
                if (key in scoreHeatmapData) {
                    scoreHeatmapData[key] += row.score;
                } else {
                    scoreHeatmapData[key] = row.score;
                }
            }
        });
        Object.keys(scoreHeatmapData).forEach(key => {
            scoreHeatmapData[key] /= ensembleMembers;
        });
        console.log(scoreHeatmapData);

        const loci = Array.from(locusSet).sort((a, b) => a - b);
        const zValues = RESIDUES.map(res =>
            loci.map(locus => {
                // If masking is enabled and this is the wild-type residue, return null
                if (res === wtResidues[locus]) {
                    if (maskWildType) {
                        return null;
                    }
                }
                const key = `${locus}-${res}`;
                // Use explicit check to handle zero values correctly
                return key in scoreHeatmapData ? scoreHeatmapData[key] : null;
            })
        );
        const zmin = showWTMarginalLikelihood ? 0 : 0;
        const zmax = showWTMarginalLikelihood ? Math.max(...zValues.flat(2).filter(val => val !== null) as number[]) : 1;

        const hoverTemplate = showWTMarginalLikelihood ? '%{customdata}%{x}%{y}<br>Score: 10^%{z}<extra></extra>' : '%{customdata}%{x}%{y}<br>Probability: %{z}<extra></extra>';

        const plotlyData: Array<Partial<Data>> = [{
            type: 'heatmap',
            z: zValues,
            x: loci,
            y: RESIDUES,
            colorscale: 'Viridis',
            hoverongaps: false,
            zmin: zmin,
            zmax: zmax,
            zauto: false,
            hovertemplate: hoverTemplate,
            customdata: RESIDUES.map(() => loci.map(locus => wtResidues[locus])),
            showscale: true,
        }];

        return (
            <div style={{ width: '100%', maxWidth: '900px' }}>
                <Plot
                    data={plotlyData}
                    layout={{
                        title: `Naturalness${showWTMarginalLikelihood ? ' (WT Marginal Likelihood, log scale)' : ' (Residue Probability)'}`,
                        xaxis: { title: 'Position in Sequence' },
                        yaxis: { title: 'Mutant Residue' },
                        height: 500,
                        autosize: true,
                        margin: { l: 50, r: 50, t: 50, b: 50 }
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '100%' }}
                />
            </div>
        );
    }, [logitCsvData, maskWildType, zeroWildType, showWTMarginalLikelihood]);

    const highlightResiduesOnModel = () => {
        if (!logitCsvData) return;
        if (!yamlConfig) {
            console.log('No yaml config, cannot highlight residues on model.');
            return;
        }
        const configHelper = new BoltzYamlHelper(yamlConfig);
        if (configHelper.getProteinSequences().length > 1) {
            notify.error('Cannot currently highlight residues on multimers.');
        }

        const tableData: RowData[] | null = parseCsvDataIntoRowData(logitCsvData, showWTMarginalLikelihood, zeroWildType, maxMutationsPerLocus, topPerformersToDisplay) || null;
        if (!tableData) return null;

        const lociToHighlight = tableData.map(row => {
            const { wtResidue, locus, mutantResidue } = parseSeqId(row.seqId);
            return locus;
        }).filter(residue => residue !== null);

        // Get unique residues to highlight
        const uniqueLociToHighlight = Array.from(new Set(lociToHighlight));

        const selection = uniqueLociToHighlight.map(locus => {
            return {
                struct_asym_id: 'A',
                start_residue_number: locus,
                end_residue_number: locus,
                color: "#FFD700",
            }
        })

        setSelectedSubsequence({
            data: selection,
            nonSelectedColor: "white",
        });
    }

    const [showHelpModal, setShowHelpModal] = useState<boolean>(false);

    return (
        <TabContainer>
            {/* Description Section */}
            <DescriptionSection title="Naturalness Overview">
                    Naturalness (TODO: describe PLMs, naturalness, logits, etc)
                    <ul>
                        <li><code>logit model</code> which PLM you want to use to predict logits</li>
                    </ul>
                    <p>
                        Once complete, you can download the "naturalness" scores for all mutants from the Files tab.
                    </p>
                    <p>
                        <code>Estimated cost:</code>~$1 per run.
                    </p>
            </DescriptionSection>

            {/* Evolution Runs Table */}
            <TableSection title="Logit Runs">
                <ResponsiveTable>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Status</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {logits?.map(logit => (
                            <tr key={logit.id}>
                                <td>{logit.name}</td>
                                <td>{getLogitStatus(logit)}</td>
                                <td>
                                    <FaFileCode
                                        uk-tooltip="View logs"
                                        onClick={() => openUpLogsForJob(logit.invokation_id || undefined)}
                                    />
                                    {
                                        getLogitStatus(logit) == 'finished' ?
                                            <>
                                                <FaEye
                                                    uk-tooltip="View results"
                                                    onClick={() => loadLogit(logit.id)} />
                                                <FaDownload
                                                    uk-tooltip="Download logit CSV."
                                                    onClick={() => downloadLogitCsv(logit)} />
                                            </> : null
                                    }
                                    <FaRedo uk-tooltip="Retry the logit run."
                                        onClick={() => rerunLogit(logit)} />
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </ResponsiveTable>
            </TableSection>

            {/* Display logit info, if requested. */}
            {
                displayedLogitId ?
                    <>
                        <CheckboxControl
                            label="Mask wild-type amino acids"
                            checked={maskWildType}
                            onChange={setMaskWildType}
                        />
                        <CheckboxControl
                            label="Zero out wild-type amino acids"
                            checked={zeroWildType}
                            onChange={setZeroWildType}
                        />
                        <CheckboxControl
                            label="Display Log(WT Marginal Likelihood)"
                            checked={showWTMarginalLikelihood}
                            onChange={setShowWTMarginalLikelihood}
                        />
                        {logitPlot}
                        <NumberInputControl
                            label="Max number of mutants per locus"
                            value={maxMutationsPerLocus}
                            onChange={setMaxMutationsPerLocus}
                            min={1}
                        />
                        <NumberInputControl
                            label="Top mutants to display"
                            value={topPerformersToDisplay}
                            onChange={setTopPerformersToDisplay}
                            min={1}
                        />
                        <LogitTable
                            logitCsvData={logitCsvData}
                            useWtMarginalAsScore={showWTMarginalLikelihood}
                            zeroWildType={zeroWildType}
                            maxMutationsPerLocus={maxMutationsPerLocus}
                            topPerformersToDisplay={topPerformersToDisplay}
                        />
                        <ButtonGroup>
                            <button
                                className="uk-button uk-button-default"
                                onClick={() => {
                                    if (!logitCsvData) return;

                                    const tableData = parseCsvDataIntoRowData(logitCsvData, showWTMarginalLikelihood, zeroWildType, maxMutationsPerLocus, topPerformersToDisplay)

                                    if (!tableData) return;

                                    const mutations = tableData
                                        .map(row => row.seqId)
                                        .join('\n');

                                    navigator.clipboard.writeText(mutations);
                                    notify.success('Mutations copied to clipboard!');
                                }}
                            >
                                Copy mutations to clipboard
                            </button>
                            <button className="uk-button uk-button-primary" onClick={() => highlightResiduesOnModel()}>
                                Highlight residues on model
                            </button>
                        </ButtonGroup>
                    </>
                    : null
            }

            {/* Collapsible New Run Section */}
            <CollapsibleSection
                title="New Naturalness Run"
                isOpen={showForm}
                onToggle={() => setShowForm(!showForm)}
            >
                {/* Help Alert */}
                <Alert
                    message="What is Naturalness?"
                    description={
                        <div>
                            <Paragraph>
                                Naturalness uses protein language models to score how "natural" each possible amino acid mutation looks.
                                Higher scores indicate mutations that are more likely to maintain protein function.
                            </Paragraph>
                            <AntButton
                                type="link"
                                icon={<QuestionCircleOutlined />}
                                onClick={() => setShowHelpModal(true)}
                                style={{ padding: 0 }}
                            >
                                View detailed naturalness guide
                            </AntButton>
                        </div>
                    }
                    type="info"
                    showIcon
                    style={{ marginBottom: '20px' }}
                />

                {/* Detailed Help Modal */}
                <Modal
                    title="Naturalness Analysis Guide"
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
                        <Title level={4}>What is Naturalness?</Title>
                        <Paragraph>
                            Naturalness analysis uses protein language models (PLMs) to evaluate how "natural" or likely
                            each possible amino acid substitution appears based on evolutionary patterns learned from
                            millions of protein sequences.
                        </Paragraph>

                        <Title level={4}>How to Use</Title>
                        <ul>
                            <li><Text strong>Model Selection:</Text> Choose from different PLMs (ESM-C models recommended)</li>
                            <li><Text strong>Structure Integration:</Text> Optionally include 3D structure information</li>
                            <li><Text strong>Depth Two Logits:</Text> Advanced option for pair mutation analysis</li>
                        </ul>

                        <Title level={4}>Interpreting Results</Title>
                        <Paragraph>
                            The heatmap shows naturalness scores for each position-residue combination:
                        </Paragraph>
                        <ul>
                            <li><Text strong>Higher scores:</Text> More "natural" mutations, likely to preserve function</li>
                            <li><Text strong>Lower scores:</Text> Less natural mutations, may disrupt protein</li>
                            <li><Text strong>Wild-type masking:</Text> Option to hide original residues for clearer visualization</li>
                        </ul>

                        <Alert
                            message="Estimated Cost"
                            description="~$1 per naturalness run"
                            type="success"
                            showIcon
                            style={{ marginTop: '16px' }}
                        />

                        <Paragraph style={{ marginTop: '16px' }}>
                            Results can be downloaded as CSV files containing naturalness scores for all single mutations.
                        </Paragraph>
                    </div>
                </Modal>

                <h3>Start New Naturalness Run</h3>
                <FormRow>
                    <FormField>
                        <TextInputControl
                            label="Name"
                            value={runName}
                            onChange={setRunName}
                        />
                    </FormField>

                    <FormField>
                        <ESMModelPicker
                            value={logitModel}
                            onChange={setLogitModel}
                        />
                    </FormField>

                    <FormField>
                        <CheckboxControl
                            label="Use Structure (experimental)"
                            checked={useStructure}
                            onChange={setUseStructure}
                        />
                    </FormField>

                    <FormField>
                        <CheckboxControl
                            label="Get Depth Two Logits (experimental)"
                            checked={getDepthTwoLogits}
                            onChange={setGetDepthTwoLogits}
                        />
                    </FormField>
                </FormRow>

                <button
                    className="uk-button uk-button-primary uk-margin-top"
                    onClick={handleStartLogit}
                    disabled={runName === ''}
                >
                    Start Logit Run
                </button>
            </CollapsibleSection>
        </TabContainer>
    );
};

export default NaturalnessTab;
