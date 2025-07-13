import React, { useState, useMemo } from 'react';
import { Logit, Invokation } from 'src/types/types';
import { FaDownload, FaEye, FaFileCode, FaRedo } from 'react-icons/fa';
import { downloadFileStraightToFilesystem, getFile } from '../../api/fileApi';
import Plot from 'react-plotly.js';
import { Data } from 'plotly.js';
import Papa from 'papaparse';
import { Selection } from './StructurePane';
import ReactDataGrid from 'react-data-grid';
import { notify } from '../../services/NotificationService';
import { BoltzYamlHelper } from '../../util/boltzYamlHelper';
import { TabContainer, DescriptionSection, TableSection, ButtonGroup } from '../../util/tabComponents';
import { AntTable, createActionButtons, defaultExpandableContent } from '../../util/AntTable';
import { CheckboxControl, NumberInputControl } from '../../util/controlComponents';
import { DataTableContainer } from '../../util/plotComponents';
import { Button as AntButton, Typography } from 'antd';
import { NaturalnessModal } from '../shared/NaturalnessModal';
import { LogitParametersModal } from '../shared/LogitParametersModal';
import { PlusOutlined } from '@ant-design/icons';



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
    const [showNaturalnessModal, setShowNaturalnessModal] = useState<boolean>(false);
    const [showParametersModal, setShowParametersModal] = useState<boolean>(false);
    const [selectedLogit, setSelectedLogit] = useState<Logit | null>(null);
    const [templateLogit, setTemplateLogit] = useState<Logit | null>(null);

    const [displayedLogitId, setDisplayedLogitId] = useState<number | null>(null);
    const [logitCsvData, setLogitCsvData] = useState<string | null>(null);
    const [maskWildType, setMaskWildType] = useState<boolean>(false);
    const [zeroWildType, setZeroWildType] = useState<boolean>(false);
    const [showWTMarginalLikelihood, setShowWTMarginalLikelihood] = useState<boolean>(true);

    const [maxMutationsPerLocus, setMaxMutationsPerLocus] = useState<number>(3);
    const [topPerformersToDisplay, setTopPerformersToDisplay] = useState<number>(24);



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
        console.log(`Downloading logits for ${logit.name} at path ${logitPath} to ${newFileName}. Do not close this window until the download is complete.`);
        downloadFileStraightToFilesystem(
            logit.fold_id,
            logitPath,
            newFileName,
            (progress: number) => {
                console.log(`Downloading ${logitPath}: ${progress}%`);
            }
        );
    };

    const viewLogitParameters = (logit: Logit) => {
        setSelectedLogit(logit);
        setShowParametersModal(true);
    };

    const redoLogit = (logit: Logit) => {
        setTemplateLogit(logit);
        setShowNaturalnessModal(true);
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
        let chainId = configHelper.getProteinSequences()[0][0];

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
                struct_asym_id: chainId,
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
            <TableSection
                title="Naturalness Runs"
                extra={
                    <AntButton
                        type="primary"
                        icon={<PlusOutlined />}
                        onClick={() => {
                            setTemplateLogit(null);
                            setShowNaturalnessModal(true);
                        }}
                    >
                        New
                    </AntButton>
                }
            >
                <AntTable<Logit>
                    dataSource={logits || []}
                    rowKey="id"
                    expandableContent={defaultExpandableContent}
                    columns={[
                        {
                            key: 'name',
                            title: 'Name',
                            dataIndex: 'name',
                        },
                        {
                            key: 'status',
                            title: 'Status',
                            render: (_, logit) => getLogitStatus(logit),
                        },
                        {
                            key: 'actions',
                            title: 'Actions',
                            width: 120,
                            render: (_, logit) => {
                                const buttons = [
                                    {
                                        icon: <FaFileCode />,
                                        onClick: () => openUpLogsForJob(logit.invokation_id || undefined),
                                        tooltip: 'View logs',
                                    },
                                    {
                                        icon: <FaRedo />,
                                        onClick: () => redoLogit(logit),
                                        tooltip: 'Redo naturalness run',
                                    },
                                ];

                                if (getLogitStatus(logit) === 'finished') {
                                    buttons.splice(1, 0, {
                                        icon: <FaEye />,
                                        onClick: () => loadLogit(logit.id),
                                        tooltip: 'View results',
                                    });
                                    buttons.splice(2, 0, {
                                        icon: <FaDownload />,
                                        onClick: () => downloadLogitCsv(logit),
                                        tooltip: 'Download logit CSV',
                                    });
                                }

                                return createActionButtons(buttons);
                            },
                        },
                    ]}
                />
            </TableSection>

            {/* Display logit info, if requested. */}
            {
                displayedLogitId ?
                    <TableSection title={""} scrollable={false}>
                        <div style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            marginBottom: "10px"
                        }}>
                            <h2 style={{ margin: 0, overflowWrap: 'anywhere' }}>
                                {logits?.find(l => l.id === displayedLogitId)?.name || "Naturalness Results"}
                            </h2>
                            <button
                                onClick={() => setDisplayedLogitId(null)}
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
                            <AntButton
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
                            </AntButton>
                            <AntButton type="primary" onClick={() => highlightResiduesOnModel()}>
                                Highlight residues on model
                            </AntButton>
                        </ButtonGroup>
                    </TableSection>
                    : null
            }


            {/* Naturalness Modal */}
            <NaturalnessModal
                key={templateLogit ? `template-${JSON.stringify(templateLogit)}` : 'new-embedding'}
                open={showNaturalnessModal}
                onClose={() => setShowNaturalnessModal(false)}
                foldIds={[foldId]}
                title={templateLogit ? "Redo Naturalness Run" : "New Naturalness Run"}
                templateNaturalnessRun={templateLogit || undefined}
            />

            {/* Parameters Modal */}
            <LogitParametersModal
                open={showParametersModal}
                onClose={() => setShowParametersModal(false)}
                logit={selectedLogit}
            />
        </TabContainer>
    );
};

export default NaturalnessTab;
