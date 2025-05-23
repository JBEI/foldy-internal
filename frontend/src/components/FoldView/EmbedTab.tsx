import React, { useState } from 'react';
import { startEmbeddings } from "../../api/embedApi";
import { Embedding, Invokation } from '../../types/types';
import { FaDownload, FaFileCode, FaRedo } from 'react-icons/fa';
import { downloadFileStraightToFilesystem } from '../../api/fileApi';
import { notify } from '../../services/NotificationService';
import { ESMModelPicker } from './ESMModelPicker';
import { TabContainer, DescriptionSection, TableSection, CollapsibleSection, FormRow, FormField, ResponsiveTable } from '../../util/tabComponents';
import { TextInputControl, TextAreaControl } from '../../util/controlComponents';

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

    return (
        <TabContainer>
            {/* Description Section */}
            <DescriptionSection title="DMS Embedding Overview">
                    This tab allows you to embed protein sequences using large language
                    models like <a href="https://github.com/evolutionaryscale/esm">ESMC</a>.
                    These embeddings can be used to do low-N directed evolution, as in the
                    Evolve tab. Each run takes in:
                    <ul>
                        <li>
                            <code>Extra Sequence IDs</code>: "WT" to embed the WT sequence, as well as other
                            variants to embed. Eg, "A43W_T67G" to embed the mutant with those two mutations.
                        </li>
                        <li>
                            <code>DMS Starting Sequence IDs</code>: For each line in this
                            field, all possible single amino acid mutants will be embedded.
                            For each input here, this produces a large number of embeddings,
                            ~19X the number of amino acids in the protein.
                        </li>
                    </ul>
                    You can embed just the wild type sequence by entering "WT" in
                    the "Extra Sequence IDs" field, as well as any other variants of interest.
                    Additionally you can get embeddings for a large number of mutants with the
                    "DMS Starting Sequence IDs" field - for each line in this field, all
                    possible single amino acid mutants will be embedded.
                    <p>
                        <code>Estimated cost:</code>~$100 for a DMS of a 500AA protein.
                    </p>
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
