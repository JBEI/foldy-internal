import React, { useState, ChangeEvent, useMemo, useEffect } from 'react';
import UIkit from 'uikit';
import { FileInfo, FewShot, Invokation, CampaignRound } from 'src/types/types';
import { deleteFewShot, runFewShot } from '../../api/fewShotApi';
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
import { TabContainer, DescriptionSection, TableSection } from '../../util/tabComponents';
import { AntTable, createActionButtons, defaultExpandableContent } from '../../util/AntTable';
import { TextInputControl, TextAreaControl, SelectControl, FileUploadControl, MultiSelectControl, NumberInputControl } from '../../util/controlComponents';
import { DataTableContainer, PlotContainer } from '../../util/plotComponents';
import FewShotMutantTable from '../shared/FewShotMutantTable';
import FewShotDebugPlots from '../shared/FewShotDebugPlots';
import FewShotModal from '../shared/FewShotModal';
import { Button as AntButton, Typography, Spin } from 'antd';
import { PlayCircleOutlined } from '@ant-design/icons';
import { getFewShotStatus } from '../../util/statusHelpers';

const { Text } = Typography;

type RowData = {
    seqId: string;
    selected: boolean | null;
    order: number | null;
    relevantMeasuredMutants: string;
    predictionMean: number;
    predictionStddev: number;
    score: number;
    modelPredictions?: number[];
}

interface FewShotTabProps {
    foldId: number;
    yamlConfig: string | null;
    jobs: Invokation[] | null;
    files: FileInfo[] | null;
    evolutions: FewShot[] | null;
    campaignRounds?: CampaignRound[] | null;
    openUpLogsForJob: (jobId?: number) => void;
    setSelectedSubsequence: (selection: Selection | null) => void;
}

const FewShotTab: React.FC<FewShotTabProps> = ({ foldId, yamlConfig, jobs, files, evolutions, campaignRounds, openUpLogsForJob, setSelectedSubsequence }) => {
    const [showFewShotModal, setShowFewShotModal] = useState<boolean>(false);
    const [fewShotTemplate, setFewShotTemplate] = useState<Partial<FewShot> | null>(null);

    const [displayedFewShotId, setDisplayedFewShotId] = useState<number | null>(null);
    const [slateBuildCsvData, setFewShotCsvData] = useState<string | null>(null);
    const [slateBuildDebugData, setFewShotDebugData] = useState<any>(null);
    const [sortOptions, setSortOptions] = useState<{ [key: string]: string[] }>({});

    // Sort evolutions (FewShot) by date_created (newest first)
    const sortedEvolutions = useMemo(() => {
        if (!evolutions) return [];
        return [...evolutions].sort((a, b) => {
            if (!a.date_created && !b.date_created) return 0;
            if (!a.date_created) return 1; // null values go to end
            if (!b.date_created) return -1;
            return new Date(b.date_created).getTime() - new Date(a.date_created).getTime();
        });
    }, [evolutions]);



    const downloadPredictedActivity = (fewShotRun: FewShot) => {
        const predictedActivityPath = fewShotRun.output_fpath_computed;
        console.log(`Downloading predicted activity for few shot run ${fewShotRun.id} at path ${predictedActivityPath}`);
        getFile(foldId, predictedActivityPath).then(
            (fileBlob: Blob) => {
                const newFname = `${fewShotRun.name}_predicted_activity.csv`;
                notify.info(`Downloading ${predictedActivityPath} with file name ${newFname}!`);
                fileDownload(fileBlob, newFname);
            },
            (e) => {
                console.log(e);
                notify.error(e.toString());
            }
        );
    };

    const rerunFewShot = async (fewShotRun: FewShot) => {
        notify.info(`Opening FewShot modal with parameters from ${fewShotRun.name}. Make sure to add the activity file, you can download the previous one from Files tab.`);
        setFewShotTemplate(fewShotRun);
        setShowFewShotModal(true);
    };

    const loadFewShot = (fewShotRunId: number) => {
        const fewShotRun = sortedEvolutions.find(fewShotRun => fewShotRun.id === fewShotRunId);
        if (!fewShotRun) {
            notify.error(`FewShot ${fewShotRunId} not found.`);
            return;
        }
        setDisplayedFewShotId(fewShotRunId);
        console.log(`Loading few shot run ${fewShotRun.name}...`);

        // Fetch the predicted activity file
        getFile(foldId, fewShotRun.output_fpath_computed).then(
            (fileBlob: Blob) => {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const fileString = e.target?.result as string;
                    setFewShotCsvData(fileString);
                };
                reader.readAsText(fileBlob);
            },
            (e) => {
                console.log(e);
                notify.error(`Error fetching predicted_activity.csv: ${e.toString()}`);
            }
        );

        // Fetch the debug.json file
        getFile(foldId, `few_shots/${fewShotRun.name}/debug_info.json`).then(
            (fileBlob: Blob) => {
                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        const fileString = e.target?.result as string;
                        // Replace NaN with null for proper JSON parsing
                        const cleanedString = fileString.replace(/NaN/g, 'null');
                        const jsonData = JSON.parse(cleanedString);
                        setFewShotDebugData(jsonData);
                        setSortOptions(jsonData.sorts);
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

    const deleteFewShotHelper = async (fewShotRunId: number) => {
        await UIkit.modal.confirm('Are you sure you want to delete this few shot run? This action is irreversible.');
        console.log(`Deleting few shot run ${fewShotRunId}...`);
        deleteFewShot(fewShotRunId).then(
            () => {
                notify.success(`FewShot ${fewShotRunId} deleted.`);
            },
            (e) => {
                notify.error(e.toString());
            }
        )
    }


    return (
        <TabContainer>
            {/* Description Section */}
            <DescriptionSection title="FewShot Runs Overview">
                This section accepts measurements of protein activity and suggests a slate of mutants for the next round of screening.
            </DescriptionSection>

            {/* FewShot Runs Table */}
            <TableSection
                title="FewShot Runs"
                extra={
                    <AntButton
                        type="primary"
                        icon={<PlayCircleOutlined />}
                        onClick={() => setShowFewShotModal(true)}
                    >
                        New
                    </AntButton>
                }
            >
                <AntTable<FewShot>
                    dataSource={sortedEvolutions}
                    rowKey="id"
                    expandableContent={defaultExpandableContent}
                    columns={[
                        {
                            key: 'name',
                            title: 'Name',
                            dataIndex: 'name',
                            ellipsis: true,
                            render: (name) => <span title={name}>{name}</span>,
                        },
                        {
                            key: 'date',
                            title: 'Date',
                            width: 80,
                            render: (_, fewShotRun) => {
                                if (!fewShotRun.date_created) return "N/A";

                                try {
                                    const date = new Date(fewShotRun.date_created);
                                    if (isNaN(date.getTime())) return "Invalid";

                                    return new Intl.DateTimeFormat('en-US', {
                                        dateStyle: "short",
                                        timeZone: "America/Los_Angeles"
                                    }).format(date);
                                } catch (error) {
                                    return "Error";
                                }
                            },
                        },
                        {
                            key: 'status',
                            title: 'Status',
                            render: (_, fewShotRun) => getFewShotStatus(fewShotRun, jobs),
                        },
                        {
                            key: 'actions',
                            title: 'Actions',
                            width: 200,
                            render: (_, fewShotRun) => {
                                const buttons = [
                                    {
                                        icon: <FaFileCode />,
                                        onClick: () => openUpLogsForJob(fewShotRun.invokation_id || undefined),
                                        tooltip: 'View logs',
                                    },
                                    {
                                        icon: <FaRedo />,
                                        onClick: () => rerunFewShot(fewShotRun),
                                        tooltip: 'Retry the few shot run',
                                    },
                                    {
                                        icon: <FaTrash />,
                                        onClick: () => deleteFewShotHelper(fewShotRun.id || 0),
                                        tooltip: 'Delete few shot run',
                                        danger: true,
                                    },
                                ];

                                if (getFewShotStatus(fewShotRun, jobs) === 'finished') {
                                    buttons.splice(1, 0, {
                                        icon: <FaEye />,
                                        onClick: () => loadFewShot(fewShotRun.id || 0),
                                        tooltip: 'View results',
                                    });
                                    buttons.splice(2, 0, {
                                        icon: <FaDownload />,
                                        onClick: () => downloadPredictedActivity(fewShotRun),
                                        tooltip: 'Download predicted activity CSV',
                                    });
                                }

                                return createActionButtons(buttons);
                            },
                        },
                    ]}
                />
            </TableSection>


            {
                displayedFewShotId ?
                    <TableSection title={""} scrollable={false}>
                        <div style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            marginBottom: "10px"
                        }}>
                            <h2 style={{ margin: 0, overflowWrap: 'anywhere' }}>
                                {sortedEvolutions.find(e => e.id === displayedFewShotId)?.name || "Slate Build Results"}
                            </h2>
                            <button
                                onClick={() => setDisplayedFewShotId(null)}
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
                            <FewShotMutantTable
                                yamlConfig={yamlConfig}
                                predictedMutantCsvData={slateBuildCsvData}
                                setSelectedSubsequence={setSelectedSubsequence}
                                sortOptions={sortOptions}
                            />
                        </div>

                        {/* Render plotly charts with the debug data */}
                        <h3>Warm Start & Training Loss</h3>
                        <FewShotDebugPlots debugData={slateBuildDebugData} />
                    </TableSection>
                    : null
            }

            {/* FewShot Modal */}
            <FewShotModal
                key={fewShotTemplate ? JSON.stringify({
                    name: fewShotTemplate.name,
                    mode: fewShotTemplate.mode,
                    num_mutants: fewShotTemplate.num_mutants,
                    embedding_files: fewShotTemplate.embedding_files,
                    naturalness_files: fewShotTemplate.naturalness_files,
                    finetuning_model_checkpoint: fewShotTemplate.finetuning_model_checkpoint,
                    few_shot_params: fewShotTemplate.few_shot_params
                }) : 'new'}
                open={showFewShotModal}
                onClose={() => {
                    setShowFewShotModal(false);
                    setFewShotTemplate(null);
                }}
                foldId={foldId}
                files={files}
                evolutions={evolutions}
                campaignRounds={campaignRounds}
                templateFewShot={fewShotTemplate}
                defaultActivityFileSource={fewShotTemplate ? 'evolution' : 'upload'}
                fewShotRunIdForActivityFile={fewShotTemplate?.id}
            />
        </TabContainer>
    );
};

export default FewShotTab;
