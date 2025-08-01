import React, { useState, useMemo } from 'react';
import { Naturalness, Invokation } from 'src/types/types';
import { FaDownload, FaEye, FaFileCode, FaRedo, FaTrash } from 'react-icons/fa';
import { downloadFileStraightToFilesystem, getFile } from '../../api/fileApi';
import { deleteNaturalness } from '../../api/embedApi';
import { Selection } from './StructurePane';
import { notify } from '../../services/NotificationService';
import { TabContainer, DescriptionSection, TableSection } from '../../util/tabComponents';
import { AntTable, createActionButtons, defaultExpandableContent } from '../../util/AntTable';
import { Button as AntButton } from 'antd';
import { NaturalnessModal } from '../shared/NaturalnessModal';
import { PlusOutlined } from '@ant-design/icons';
import NaturalnessResults from '../shared/NaturalnessResults';
import { getNaturalnessStatus } from '../../util/statusHelpers';




interface NaturalnessTabProps {
    foldId: number;
    foldName: string | null;
    yamlConfig: string | null;
    jobs: Invokation[] | null;
    logits: Naturalness[] | null;
    setSelectedSubsequence: (selection: Selection | null) => void;
    openUpLogsForJob: (jobId: number | undefined) => void;
}

const NaturalnessTab: React.FC<NaturalnessTabProps> = ({ foldId, foldName, yamlConfig, jobs, logits, setSelectedSubsequence, openUpLogsForJob }) => {
    const [showNaturalnessModal, setShowNaturalnessModal] = useState<boolean>(false);
    const [templateNaturalness, setTemplateNaturalness] = useState<Naturalness | null>(null);

    const [displayedNaturalnessId, setDisplayedNaturalnessId] = useState<number | null>(null);
    const [naturalnessCsvData, setNaturalnessCsvData] = useState<string | null>(null);

    // Sort naturalness runs by date_created (newest first)
    const sortedLogits = useMemo(() => {
        if (!logits) return [];
        return [...logits].sort((a, b) => {
            if (!a.date_created && !b.date_created) return 0;
            if (!a.date_created) return 1; // null values go to end
            if (!b.date_created) return -1;
            return new Date(b.date_created).getTime() - new Date(a.date_created).getTime();
        });
    }, [logits]);




    const downloadNaturalnessCsv = (naturalness: Naturalness) => {
        if (!foldName) {
            notify.warning('Fold name is not set.');
            return;
        }
        const naturalnessPath = naturalness.output_fpath_computed;
        const newFileName = `${foldName}_naturalness_${naturalness.name}.csv`;
        console.log(`Downloading naturalness for ${naturalness.name} at path ${naturalnessPath} to ${newFileName}. Do not close this window until the download is complete.`);
        downloadFileStraightToFilesystem(
            naturalness.fold_id,
            naturalnessPath,
            newFileName,
            (progress: number) => {
                console.log(`Downloading ${naturalnessPath}: ${progress}%`);
            }
        );
    };


    const redoNaturalness = (naturalness: Naturalness) => {
        setTemplateNaturalness(naturalness);
        setShowNaturalnessModal(true);
    };

    const deleteNaturalnessHelper = async (naturalness: Naturalness) => {
        if (!window.confirm(`Are you sure you want to delete naturalness run "${naturalness.name}"? This action cannot be undone.`)) {
            return;
        }

        try {
            await deleteNaturalness(naturalness.id);
            notify.success(`Naturalness run "${naturalness.name}" deleted successfully`);
            window.location.reload(); // Refresh the page to update the data
        } catch (error: any) {
            notify.error(error.response?.data?.message || 'Failed to delete naturalness run');
        }
    };

    const loadNaturalness = (naturalnessId: number) => {
        const naturalness = sortedLogits.find(naturalness => naturalness.id === naturalnessId);
        if (!naturalness) {
            notify.error(`Naturalness ${naturalnessId} not found.`);
            return;
        }
        setDisplayedNaturalnessId(naturalnessId);
        console.log(`Loading naturalness ${naturalness.name}...`);

        getFile(foldId, naturalness.output_fpath_computed).then(
            (fileBlob: Blob) => {
                // Create a FileReader to read the blob as text
                const reader = new FileReader();
                reader.onload = (e) => {
                    const fileString = e.target?.result as string;
                    setNaturalnessCsvData(fileString);
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
                            setTemplateNaturalness(null);
                            setShowNaturalnessModal(true);
                        }}
                    >
                        New
                    </AntButton>
                }
            >
                <AntTable<Naturalness>
                    dataSource={sortedLogits}
                    rowKey="id"
                    expandableContent={defaultExpandableContent}
                    columns={[
                        {
                            key: 'name',
                            title: 'Name',
                            dataIndex: 'name',
                        },
                        {
                            key: 'date',
                            title: 'Date',
                            width: 80,
                            render: (_, naturalness) => {
                                if (!naturalness.date_created) return "N/A";

                                try {
                                    const date = new Date(naturalness.date_created);
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
                            render: (_, naturalness) => getNaturalnessStatus(naturalness, jobs),
                        },
                        {
                            key: 'actions',
                            title: 'Actions',
                            width: 120,
                            render: (_, naturalness) => {
                                const buttons = [
                                    {
                                        icon: <FaFileCode />,
                                        onClick: () => openUpLogsForJob(naturalness.invokation_id || undefined),
                                        tooltip: 'View logs',
                                    },
                                    {
                                        icon: <FaRedo />,
                                        onClick: () => redoNaturalness(naturalness),
                                        tooltip: 'Redo naturalness run',
                                    },
                                ];

                                if (getNaturalnessStatus(naturalness, jobs) === 'finished') {
                                    buttons.splice(1, 0, {
                                        icon: <FaEye />,
                                        onClick: () => loadNaturalness(naturalness.id),
                                        tooltip: 'View results',
                                    });
                                    buttons.splice(2, 0, {
                                        icon: <FaDownload />,
                                        onClick: () => downloadNaturalnessCsv(naturalness),
                                        tooltip: 'Download naturalness CSV',
                                    });
                                }

                                // Add delete button (always available)
                                buttons.push({
                                    icon: <FaTrash />,
                                    onClick: () => deleteNaturalnessHelper(naturalness),
                                    tooltip: 'Delete naturalness run',
                                    danger: true,
                                });

                                return createActionButtons(buttons);
                            },
                        },
                    ]}
                />
            </TableSection>

            {/* Display naturalness info, if requested. */}
            {
                displayedNaturalnessId && naturalnessCsvData ?
                    <TableSection title={""} scrollable={false}>
                        <NaturalnessResults
                            naturalnessCsvData={naturalnessCsvData}
                            yamlConfig={yamlConfig}
                            setSelectedSubsequence={setSelectedSubsequence}
                            runName={sortedLogits.find(l => l.id === displayedNaturalnessId)?.name || "Naturalness Results"}
                            onClose={() => setDisplayedNaturalnessId(null)}
                        />
                    </TableSection>
                    : null
            }


            {/* Naturalness Modal */}
            <NaturalnessModal
                key={templateNaturalness ? `template-${JSON.stringify(templateNaturalness)}` : 'new-naturalness'}
                open={showNaturalnessModal}
                onClose={() => setShowNaturalnessModal(false)}
                foldIds={[foldId]}
                title={templateNaturalness ? "Redo Naturalness Run" : "New Naturalness Run"}
                templateNaturalnessRun={templateNaturalness || undefined}
            />

        </TabContainer>
    );
};

export default NaturalnessTab;
