import React, { useState } from 'react';
import { startEmbeddings } from "../../api/embedApi";
import { Embedding, Invokation } from '../../types/types';
import { FaDownload, FaFileCode, FaRedo } from 'react-icons/fa';
import { downloadFileStraightToFilesystem } from '../../api/fileApi';
import { notify } from '../../services/NotificationService';
import { ESMModelPicker } from './ESMModelPicker';
import { TabContainer, DescriptionSection, TableSection, CollapsibleSection, FormRow, FormField, ResponsiveTable } from '../../util/tabComponents';
import { TextInputControl, TextAreaControl } from '../../util/controlComponents';
import { Alert, Modal, Button as AntButton, Typography } from 'antd';
import { QuestionCircleOutlined } from '@ant-design/icons';

const { Text, Paragraph, Title } = Typography;

interface EmbedTabProps {
    foldId: number;
    foldName: string | null;
    jobs: Invokation[] | null;
    embeddings: Embedding[] | null;
    openUpLogsForJob: (jobId: number | undefined) => void;
}

const EmbedTab: React.FC<EmbedTabProps> = ({ foldId, foldName, jobs, embeddings, openUpLogsForJob }) => {
    const [batchName, setBatchName] = useState<string | null>(null);
    const [dmsStartingSeqIds, setDmsStartingSeqIds] = useState<string>('WT');
    const [extraSequenceIDs, setExtraSequenceIDs] = useState<string>('');
    const [extraLayers, setExtraLayers] = useState<string>('');
    const [showEmbeddingSection, setShowEmbeddingSection] = useState<boolean>(false);
    const [model, setModel] = useState<string>('esmc_300m');


    const handleStartDmsEmbeddings = async () => {
        const dmsStartingSeqIdsArray: string[] = dmsStartingSeqIds
            .split('\n')
            .map(line => line.trim())
            .filter(line => line !== '');
        const extraIDsArray: string[] = extraSequenceIDs
            .split('\n')
            .map(line => line.trim())
            .filter(line => line !== '');
        const extraLayersArray: string[] = extraLayers
            .split(',')
            .map(line => line.trim())
            .filter(line => line !== '');

        if (!batchName) {
            notify.error('Batch name is required.');
            return;
        }

        try {
            await startEmbeddings(foldId, batchName, dmsStartingSeqIdsArray, extraIDsArray, extraLayersArray, model);
            notify.success('Started embedding run.');
        } catch (error) {
            notify.error(`Failed to start embedding run: ${error}`);
        }
    };

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

    const rerunEmbedding = async (embedding: Embedding) => {
        notify.info(`Repopulating "New Embedding Run" with parameters from ${embedding.name}.`);
        console.log(embedding);
        setBatchName(embedding.name);
        setDmsStartingSeqIds(embedding.dms_starting_seq_ids?.split(',').join('\n') || '');
        setExtraSequenceIDs(embedding.extra_seq_ids?.split(',').join('\n') || '');
        setExtraLayers(embedding.extra_layers?.split(',').join(',') || '');
        setShowEmbeddingSection(true);
        setModel(embedding.embedding_model);
    };

    const [showHelpModal, setShowHelpModal] = useState<boolean>(false);

    return (
        <TabContainer>
            {/* Description Section */}
            <DescriptionSection title="Protein Embeddings Overview">
                Generate high-dimensional vector representations of protein sequences using large language
                models like <a href="https://github.com/evolutionaryscale/esm">ESMC</a>.
                These embeddings can be used for machine learning models in directed evolution.
            </DescriptionSection>

            {/* Batch Status Section */}
            <TableSection title="Ongoing Batches">
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
                                        uk-tooltip="Rerun embedding."
                                        onClick={() => rerunEmbedding(embedding)} />
                                </td>
                            </tr>
                        ))
                            || <tr><td colSpan={2}>No embeddings available</td></tr>}
                    </tbody>
                </ResponsiveTable>
            </TableSection>

            {/* Collapsible Section */}
            <CollapsibleSection
                title="New Embedding Run"
                isOpen={showEmbeddingSection}
                onToggle={() => setShowEmbeddingSection(!showEmbeddingSection)}
            >
                {/* Help Alert */}
                <Alert
                    message="What are Protein Embeddings?"
                    description={
                        <div>
                            <Paragraph>
                                Generate high-dimensional vector representations of protein sequences using large language models.
                                These embeddings capture structural and functional information for use in machine learning models.
                            </Paragraph>
                            <AntButton
                                type="link"
                                icon={<QuestionCircleOutlined />}
                                onClick={() => setShowHelpModal(true)}
                                style={{ padding: 0 }}
                            >
                                View detailed embedding guide
                            </AntButton>
                        </div>
                    }
                    type="info"
                    showIcon
                    style={{ marginBottom: '20px' }}
                />

                {/* Detailed Help Modal */}
                <Modal
                    title="Protein Embedding Guide"
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
                        <Title level={4}>What are Protein Embeddings?</Title>
                        <Paragraph>
                            Protein embeddings are high-dimensional vector representations that capture the structural
                            and functional properties of protein sequences. Generated using large language models like
                            <a href="https://github.com/evolutionaryscale/esm" target="_blank" rel="noopener noreferrer"> ESMC</a>,
                            these embeddings can be used for downstream machine learning tasks.
                        </Paragraph>

                        <Title level={4}>Input Types Explained</Title>
                        <Paragraph>
                            There are two main input types for embedding generation:
                        </Paragraph>
                        <ul>
                            <li>
                                <Text strong>Extra Sequence IDs:</Text> Specific variants you want to embed individually:
                                <ul style={{ marginTop: '8px', marginLeft: '16px' }}>
                                    <li>"WT" - embeds the wild-type sequence</li>
                                    <li>"A43W_T67G" - embeds a specific double mutant</li>
                                    <li>"K150R" - embeds a single point mutation</li>
                                    <li>Use one sequence ID per line</li>
                                </ul>
                            </li>
                            <li style={{ marginTop: '12px' }}>
                                <Text strong>DMS Starting Sequence IDs:</Text> Base sequences for comprehensive mutational scanning:
                                <ul style={{ marginTop: '8px', marginLeft: '16px' }}>
                                    <li>For each sequence listed, ALL possible single amino acid mutants will be generated and embedded</li>
                                    <li>Creates ~19× the protein length in embeddings (19 amino acids × each position)</li>
                                    <li>Example: "WT" will generate embeddings for A1C, A1D, A1E... through to the last position</li>
                                    <li>Use this for deep mutational scanning experiments</li>
                                </ul>
                            </li>
                        </ul>

                        <Alert
                            message="Example Usage"
                            description={
                                <div>
                                    <Text strong>Extra Sequence IDs:</Text> WT, A43W_T67G, K150R<br />
                                    <Text strong>DMS Starting Sequence IDs:</Text> WT<br />
                                    <Text>This will embed the wild-type, two specific mutants, plus all single mutants of the wild-type.</Text>
                                </div>
                            }
                            type="success"
                            showIcon
                            style={{ marginTop: '12px', marginBottom: '12px' }}
                        />

                        <Title level={4}>Model Selection</Title>
                        <Paragraph>
                            Choose a pLM model:
                        </Paragraph>
                        <ul>
                            <li><Text strong>ESMC 300M:</Text> Fast, what was evaluated in the FolDE paper. Available for academic use.</li>
                            <li><Text strong>ESM2 15B:</Text> Slower, used in the EvolvePro paper..</li>
                        </ul>

                        <Title level={4}>Workflow Integration</Title>
                        <Paragraph>
                            The typical workflow for using embeddings in directed evolution:
                        </Paragraph>
                        <ol>
                            <li><Text strong>Generate embeddings:</Text> Create embeddings for your protein variants (this tab)</li>
                            <li><Text strong>Collect activity data:</Text> Measure experimental activity for some variants</li>
                            <li><Text strong>Train models:</Text> Use embeddings + activity data in the Evolution tab</li>
                            <li><Text strong>Predict activities:</Text> Get recommendations for the most promising mutations</li>
                        </ol>

                        <Title level={4}>Best Practices</Title>
                        <ul>
                            <li><Text strong>Start small:</Text> Test with a few variants first before running full DMS</li>
                            <li><Text strong>Include controls:</Text> Always include "WT" in your Extra Sequence IDs</li>
                            <li><Text strong>Model selection:</Text> ESMC 600M offers good balance of speed and quality</li>
                            <li><Text strong>Plan ahead:</Text> DMS generates many embeddings - ensure you need them all</li>
                        </ul>

                        <Alert
                            message="Cost Consideration"
                            description="~$100 for a DMS of a 500AA protein. Consider using Extra Sequence IDs for targeted studies."
                            type="warning"
                            showIcon
                            style={{ marginTop: '16px' }}
                        />

                        <Paragraph style={{ marginTop: '16px' }}>
                            <Text strong>Output:</Text> Completed embeddings can be downloaded as CSV files from the Files tab and are automatically available for use in the Evolution tab.
                        </Paragraph>
                    </div>
                </Modal>

                <h4>Start a New Embedding Run</h4>
                <TextInputControl
                    label="Batch Name"
                    value={batchName || ''}
                    onChange={setBatchName}
                    placeholder="Enter batch name"
                />

                <FormRow>
                    <FormField>
                        <TextAreaControl
                            label="Extra Sequence IDs"
                            value={extraSequenceIDs}
                            onChange={setExtraSequenceIDs}
                            placeholder="Enter one mutation per line, e.g., A37T, W100C_T431G"
                        />
                    </FormField>

                    <FormField>
                        <TextAreaControl
                            label="DMS Starting Sequence IDs"
                            value={dmsStartingSeqIds}
                            onChange={setDmsStartingSeqIds}
                            placeholder="Enter one mutation per line, e.g., WT, W100C_T431G"
                        />
                    </FormField>
                </FormRow>

                <ESMModelPicker
                    value={model}
                    onChange={setModel}
                />

                <TextAreaControl
                    label="Extra Layers"
                    value={extraLayers}
                    onChange={setExtraLayers}
                    placeholder="Enter extra embedding layers to extract like 5,10,15"
                    rows={1}
                    inputStyle={{ resize: 'vertical' }}
                />

                <div style={{ marginTop: '20px' }}>
                    <button
                        className="uk-button uk-button-primary"
                        onClick={() => handleStartDmsEmbeddings()}
                    >
                        Start Embedding
                    </button>
                </div>
            </CollapsibleSection>
        </TabContainer>
    );
};

export default EmbedTab;
