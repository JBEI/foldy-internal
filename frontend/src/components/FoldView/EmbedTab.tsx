import React, { useState } from 'react';
import { Embedding, Invokation } from '../../types/types';
import { FaDownload, FaFileCode, FaRedo } from 'react-icons/fa';
import { downloadFileStraightToFilesystem } from '../../api/fileApi';
import { notify } from '../../services/NotificationService';
import { TabContainer, DescriptionSection, TableSection } from '../../util/tabComponents';
import { AntTable, createActionButtons } from '../../util/AntTable';
import { Button as AntButton, Table } from 'antd';
import { EmbeddingModal } from '../shared/EmbeddingModal';
import { EmbeddingParametersModal } from '../shared/EmbeddingParametersModal';
import { PlusOutlined } from '@ant-design/icons';

interface EmbedTabProps {
    foldId: number;
    foldName: string | null;
    jobs: Invokation[] | null;
    embeddings: Embedding[] | null;
    openUpLogsForJob: (jobId: number | undefined) => void;
}

// Custom expandable content for embeddings with text wrapping and line limits
const embeddingExpandableContent = <T extends Record<string, any>>(record: T): React.ReactNode => {
    const entries = Object.entries(record).filter(([key, value]) =>
        value !== null && value !== undefined && value !== ''
    );

    const detailColumns = [
        {
            title: 'Property',
            dataIndex: 'key',
            key: 'key',
            width: 200,
            render: (key: string) => <strong>{key}</strong>,
        },
        {
            title: 'Value',
            dataIndex: 'value',
            key: 'value',
            render: (value: any) => {
                if (typeof value === 'object') {
                    return <pre style={{ margin: 0, fontSize: '12px' }}>{JSON.stringify(value, null, 2)}</pre>;
                }
                if (typeof value === 'boolean') {
                    return value ? 'true' : 'false';
                }

                const stringValue = String(value);
                // For long strings, apply wrapping and height constraints
                if (stringValue.length > 100) {
                    return (
                        <div style={{
                            maxHeight: '5rem', // Approximately 5 lines
                            overflowY: 'auto',
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word',
                            fontSize: '12px',
                            lineHeight: '1rem',
                            padding: '4px',
                            backgroundColor: '#f5f5f5',
                            border: '1px solid #d9d9d9',
                            borderRadius: '4px'
                        }}>
                            {stringValue}
                        </div>
                    );
                }

                return stringValue;
            },
        },
    ];

    const detailData = entries.map(([key, value]) => ({
        key,
        value,
    }));

    return (
        <Table
            columns={detailColumns}
            dataSource={detailData}
            pagination={false}
            size="small"
            bordered
            rowKey="key"
            style={{ margin: '16px 0' }}
        />
    );
};

const EmbedTab: React.FC<EmbedTabProps> = ({ foldId, foldName, jobs, embeddings, openUpLogsForJob }) => {
    const [showEmbeddingModal, setShowEmbeddingModal] = useState<boolean>(false);
    const [selectedEmbedding, setSelectedEmbedding] = useState<Embedding | null>(null);
    const [templateEmbedding, setTemplateEmbedding] = useState<Embedding | null>(null);



    const getEmbeddingStatus = (embedding: Embedding): string => {
        const job = jobs?.find(job => job.id === embedding.invokation_id);
        return job?.state || 'Unknown';
    };

    const downloadEmbedding = (embedding: Embedding) => {
        const paddedFoldId = foldId.toString().padStart(6, '0');
        const embeddingPath = `embed/${paddedFoldId}_embeddings_${embedding.embedding_model}_${embedding.name}.csv`;
        notify.info(`Downloading embedding ${embedding.id} at path ${embeddingPath}, do not close this window until the download is complete.`);

        const newFileName = `${foldName || paddedFoldId}_embedding_${embedding.name}.csv`;
        downloadFileStraightToFilesystem(embedding.fold_id, embeddingPath, newFileName, (progress: number) => {
            console.log(`Downloading ${embeddingPath}: ${progress}%`);
        });
    };

    const redoEmbedding = (embedding: Embedding) => {
        setTemplateEmbedding(embedding);
        setShowEmbeddingModal(true);
    };

    return (
        <TabContainer>
            {/* Description Section */}
            <DescriptionSection title="Protein Embeddings Overview">
                Generate high-dimensional vector representations of protein sequences using large language
                models like <a href="https://github.com/evolutionaryscale/esm">ESMC</a>.
                These embeddings can be used for machine learning models in directed evolution.
            </DescriptionSection>

            {/* Batch Status Section */}
            <TableSection
                title="Embedding Runs"
                extra={
                    <AntButton
                        type="primary"
                        icon={<PlusOutlined />}
                        onClick={() => {
                            setTemplateEmbedding(null);
                            setShowEmbeddingModal(true);
                        }}
                    >
                        New
                    </AntButton>
                }
            >
                <AntTable<Embedding>
                    dataSource={embeddings || []}
                    rowKey="id"
                    expandableContent={embeddingExpandableContent}
                    columns={[
                        {
                            key: 'name',
                            title: 'Batch Name',
                            dataIndex: 'name',
                        },
                        {
                            key: 'status',
                            title: 'Batch Status',
                            render: (_, embedding) => getEmbeddingStatus(embedding),
                        },
                        {
                            key: 'actions',
                            title: 'Actions',
                            width: 120,
                            render: (_, embedding) => {
                                const buttons = [
                                    {
                                        icon: <FaFileCode />,
                                        onClick: () => openUpLogsForJob(embedding.invokation_id || undefined),
                                        tooltip: 'View logs',
                                    },
                                    {
                                        icon: <FaRedo />,
                                        onClick: () => redoEmbedding(embedding),
                                        tooltip: 'Redo embedding run',
                                    },
                                ];

                                if (getEmbeddingStatus(embedding) === 'finished') {
                                    buttons.splice(1, 0, {
                                        icon: <FaDownload />,
                                        onClick: () => downloadEmbedding(embedding),
                                        tooltip: 'Download embeddings CSV',
                                    });
                                }

                                return createActionButtons(buttons);
                            },
                        },
                    ]}
                />
            </TableSection>


            {/* Embedding Modal */}
            <EmbeddingModal
                key={templateEmbedding ? `template-${JSON.stringify(templateEmbedding)}` : 'new-embedding'}
                open={showEmbeddingModal}
                onClose={() => setShowEmbeddingModal(false)}
                foldIds={[foldId]}
                title={templateEmbedding ? "Redo Embedding Run" : "New Embedding Run"}
                templateEmbedding={templateEmbedding || undefined}
            />
        </TabContainer>
    );
};

export default EmbedTab;
