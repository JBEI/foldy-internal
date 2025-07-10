import React, { useState } from 'react';
import { Embedding, Invokation } from '../../types/types';
import { FaDownload, FaFileCode, FaRedo } from 'react-icons/fa';
import { downloadFileStraightToFilesystem } from '../../api/fileApi';
import { notify } from '../../services/NotificationService';
import { TabContainer, DescriptionSection, TableSection, ResponsiveTable } from '../../util/tabComponents';
import { Button as AntButton } from 'antd';
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

const EmbedTab: React.FC<EmbedTabProps> = ({ foldId, foldName, jobs, embeddings, openUpLogsForJob }) => {
    const [showEmbeddingModal, setShowEmbeddingModal] = useState<boolean>(false);
    const [showParametersModal, setShowParametersModal] = useState<boolean>(false);
    const [selectedEmbedding, setSelectedEmbedding] = useState<Embedding | null>(null);
    const [templateEmbedding, setTemplateEmbedding] = useState<Embedding | null>(null);



    const getEmbeddingStatus = (embedding: Embedding): string => {
        const job = jobs?.find(job => job.id === embedding.invokation_id);
        return job?.state || 'Unknown';
    };

    const downloadEmbedding = (embedding: Embedding) => {
        const paddedFoldId = foldId.toString().padStart(6, '0');
        const embeddingPath = `embed/${paddedFoldId}_embeddings_${embedding.embedding_model}_${embedding.name}.csv`;
        notify.info(`Downloading embedding ${embedding.id} at path ${embeddingPath}`);

        const newFileName = `${foldName || paddedFoldId}_embedding_${embedding.name}.csv`;
        downloadFileStraightToFilesystem(embedding.fold_id, embeddingPath, newFileName, (progress: number) => {
            console.log(`Downloading ${embeddingPath}: ${progress}%`);
        });
    };

    const viewEmbeddingParameters = (embedding: Embedding) => {
        setSelectedEmbedding(embedding);
        setShowParametersModal(true);
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
                <ResponsiveTable>
                    <thead>
                        <tr>
                            <th>Batch Name</th>
                            <th>Batch Status</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {embeddings?.map(embedding => (
                            <tr key={embedding.id}>
                                <td>{embedding.name}</td>
                                <td>{getEmbeddingStatus(embedding)}</td>
                                <td>

                                    <FaFileCode
                                        uk-tooltip="View logs"
                                        onClick={() => openUpLogsForJob(embedding.invokation_id || undefined)}
                                    />
                                    {
                                        getEmbeddingStatus(embedding) == 'finished' ?
                                            <FaDownload
                                                uk-tooltip="Download embeddings CSV."
                                                onClick={() => downloadEmbedding(embedding)} />
                                            : null
                                    }
                                    <FaRedo
                                        uk-tooltip="Redo embedding run"
                                        onClick={() => redoEmbedding(embedding)} />
                                </td>
                            </tr>
                        ))
                            || <tr><td colSpan={2}>No embeddings available</td></tr>}
                    </tbody>
                </ResponsiveTable>
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

            {/* Parameters Modal */}
            <EmbeddingParametersModal
                open={showParametersModal}
                onClose={() => setShowParametersModal(false)}
                embedding={selectedEmbedding}
            />
        </TabContainer>
    );
};

export default EmbedTab;
