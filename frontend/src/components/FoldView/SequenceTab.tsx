import React, { useState } from "react";
import { EditableTagList } from "../../util/editableTagList";
import SeqViz from "seqviz";
import { AiFillEdit } from "react-icons/ai";
import { BoltzYamlHelper, ChainSequence, LigandData } from "../../util/boltzYamlHelper";
import BoltzYamlBuilder from "../../util/boltzYamlBuilder";
import UIkit from "uikit";
import { Selection } from "./StructurePane";
import { notify } from "../../services/NotificationService";
import { TabContainer, SectionCard, CollapsibleSection, FormRow, FormField } from "../../util/tabComponents";
import { CheckboxControl } from "../../util/controlComponents";
import { Alert, Modal, Button as AntButton, Typography, Form, Input, Switch, Tag } from 'antd';
import { QuestionCircleOutlined, EditOutlined } from '@ant-design/icons';

const { Text, Paragraph, Title } = Typography;

export interface SubsequenceSelection {
    chainIdx: number;
    startResidue: number;
    endResidue: number;
    subsequence: string;
}

interface SequenceTabProps {
    foldId: number;
    foldName: string;
    foldTags: string[];
    foldOwner: string;
    foldCreateDate: string;
    foldPublic: boolean | null;
    yamlConfig: string | null;
    foldDiffusionSamples: number | null;

    // Old AlphaFold inputs.
    sequence: string | null;
    foldModelPreset: string | null;
    foldDisableRelaxation: boolean | null;

    colorScheme: string;

    setPublic: (is_public: boolean) => void;
    setDisableRelaxation: (disable_relaxation: boolean) => void;
    setFoldName: () => void;
    setFoldModelPreset: () => void;
    addTag: (tagToAdd: string) => void;
    deleteTag: (tagToDelete: string) => void;
    handleTagClick: (tagToOpen: string) => void;

    setSelectedSubsequence: (selection: Selection | null) => void;

    userType: string | null;
    setYamlConfig: (yaml: string) => void;
}

const SequenceTab = React.memo((props: SequenceTabProps) => {
    const [showYamlSection, setShowYamlSection] = useState<boolean>(false);

    const configHelper = props.yamlConfig ? new BoltzYamlHelper(props.yamlConfig) : null;

    var sequenceNames: string[];
    var sequences: string[];
    if (configHelper) {
        sequenceNames = configHelper.getProteinSequences().map((e) => e[0]);
        sequences = configHelper.getProteinSequences().map((e) => e[1]);
    } else if (props.sequence) {
        const oldSequenceStrs = props.sequence.split(";");
        sequenceNames = oldSequenceStrs.map((ss) => ss.includes(":") ? ss.split(":")[0] : props.foldName);
        sequences = oldSequenceStrs.map((ss) => ss.includes(":") ? ss.split(":")[1] : ss);
    } else {
        return <div>No sequence found.</div>
    }

    const renderSequenceViewer = () => {
        return <>
            {sequences.map((ss: string, idx: number) => {
                const chainName = sequenceNames[idx];
                const chainSeq = ss;

                const onSelectionHandler = (chainName: string, selection: any) => {
                    if (!configHelper) {
                        console.log('Config helper is underfined, cannot show residues.')
                        return;
                    }
                    const chainIndex = configHelper.getProteinSequences().findIndex(x => x[0] === chainName);
                    if (chainIndex === -1) {
                        console.log(`Could not find chain ${chainName} in boltz config: ${configHelper?.getProteinSequences()}`);
                        return;
                    }
                    const chainId = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[chainIndex]
                    if (selection.start && selection.end) {
                        console.log(selection);
                        var start = Math.min(selection.start, selection.end);
                        var end = Math.max(selection.start, selection.end);
                        if (start >= end) {
                            start = -1;
                            end = 0;
                        }
                        console.log(`${start}, ${end}`)
                        props.setSelectedSubsequence({
                            data: [{
                                struct_asym_id: chainId,
                                start_residue_number: start + 1,
                                end_residue_number: end,
                                color: "white",
                            }],
                            // nonSelectedColor: "white",
                        });
                    }
                };

                return (
                    <div key={`sequence-${chainName}-${idx}`} style={{ marginBottom: "20px" }}>
                        <h3>{chainName}</h3>
                        <SeqViz
                            key={`seqviz-${chainName}-${idx}`}
                            name={chainName}
                            seq={chainSeq}
                            seqType="aa"
                            viewer="linear"
                            showComplement={false}
                            zoom={{ linear: 10 }}
                            style={{
                                width: "100%",
                                marginBottom: "20px",
                                border: "1px solid #e0e0e0",
                                borderRadius: "8px",
                            }}
                            onSelection={(selection: any) => onSelectionHandler(chainName, selection)}
                        />
                    </div>
                );
            })}
            {configHelper?.getLigands().map((ligand: LigandData, idx: number) => {
                return <div key={idx} style={{ marginBottom: "20px" }}>
                    <h3>{ligand.chain_ids.join(", ")} (Ligand)</h3>
                    <div>
                        {ligand.smiles || ligand.ccd}
                    </div>
                </div>
            })}
            {configHelper?.getDNASequences().map((dna: ChainSequence, idx: number) => {
                return <div key={idx} style={{ marginBottom: "20px" }}>
                    <h3>{dna[0]} (DNA)</h3>
                    <div>
                        <SeqViz
                            name={dna[0]}
                            seq={dna[1]}
                            seqType="dna"
                            viewer="linear"
                            style={{
                                width: "100%",
                                marginBottom: "20px",
                                border: "1px solid #e0e0e0",
                                borderRadius: "8px",
                            }}
                        />
                    </div>
                </div>
            })}
            {configHelper?.getRNASequences().map((rna: ChainSequence, idx: number) => {
                return <div key={idx} style={{ marginBottom: "20px" }}>
                    <h3>{rna[0]} (RNA)</h3>
                    <div>
                        <SeqViz
                            name={rna[0]}
                            seq={rna[1]}
                            seqType="rna"
                            viewer="linear"
                            style={{
                                width: "100%",
                                marginBottom: "20px",
                                border: "1px solid #e0e0e0",
                                borderRadius: "8px",
                            }}
                        />
                        {rna[1]}
                    </div>
                </div>
            })}
        </>
    };

    const canEditYaml = props.userType !== "viewer";

    const [showHelpModal, setShowHelpModal] = useState<boolean>(false);

    return (
        <TabContainer>
            {/* Sequence Viewer */}
            <SectionCard>
                {renderSequenceViewer()}
            </SectionCard>

            {/* YAML Builder Section - only show if user has permission */}
            {canEditYaml && (
                <CollapsibleSection
                    title="Edit YAML Configuration"
                    isOpen={showYamlSection}
                    onToggle={() => setShowYamlSection(!showYamlSection)}
                    style={{ marginBottom: '20px' }}
                >
                    <BoltzYamlBuilder
                        initialYaml={props.yamlConfig || undefined}
                        onSave={(yaml) => {
                            console.log(`YAML: ${yaml}`);
                            UIkit.modal
                                .confirm(
                                    `Are you sure you want to update the YAML configuration?`
                                )
                                .then(async () => {
                                    await props.setYamlConfig(yaml);
                                    notify.info("Updated YAML configuration. You can refold the protein from Actions > Refold.");
                                });
                        }}
                    />
                </CollapsibleSection>
            )}

            {/* Form Section */}
            <SectionCard>
                {/* Help Alert */}
                <Alert
                    message="Protein Properties & Settings"
                    description={
                        <div>
                            <Paragraph>
                                Configure basic protein properties, tags, and folding parameters.
                                Use the edit buttons to modify name and model preset settings.
                            </Paragraph>
                            <AntButton
                                type="link"
                                icon={<QuestionCircleOutlined />}
                                onClick={() => setShowHelpModal(true)}
                                style={{ padding: 0 }}
                            >
                                View detailed property guide
                            </AntButton>
                        </div>
                    }
                    type="info"
                    showIcon
                    style={{ marginBottom: '20px' }}
                />

                {/* Detailed Help Modal */}
                <Modal
                    title="Protein Properties Guide"
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
                        <Title level={4}>Basic Properties</Title>
                        <ul>
                            <li><Text strong>Name:</Text> Descriptive identifier for your protein</li>
                            <li><Text strong>Owner:</Text> User who created this fold</li>
                            <li><Text strong>Created:</Text> Timestamp of fold creation</li>
                            <li><Text strong>Diffusion Samples:</Text> Number of samples used in structure generation</li>
                        </ul>

                        <Title level={4}>Visibility & Organization</Title>
                        <ul>
                            <li><Text strong>Public:</Text> Make this fold visible to other users</li>
                            <li><Text strong>Tags:</Text> Add descriptive labels for organization and search</li>
                        </ul>

                        <Title level={4}>Folding Parameters</Title>
                        <ul>
                            <li><Text strong>Model Preset:</Text> Structure prediction algorithm configuration</li>
                            <li><Text strong>Disable Relaxation:</Text> Skip energy minimization step (faster but less refined)</li>
                        </ul>

                        <Alert
                            message="Parameter Changes"
                            description="Changes to folding parameters require refolding the protein to take effect."
                            type="warning"
                            showIcon
                            style={{ marginTop: '16px' }}
                        />
                    </div>
                </Modal>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px 24px', maxWidth: '800px' }}>
                    {/* Row 1 */}
                    <div>
                        <Text strong style={{ display: 'block', marginBottom: '4px' }}>Name</Text>
                        <Input.Group compact>
                            <Input
                                value={props.foldName}
                                disabled
                                style={{ width: 'calc(100% - 40px)' }}
                                size="small"
                            />
                            <AntButton
                                icon={<EditOutlined />}
                                onClick={props.setFoldName}
                                disabled={props.userType === "viewer"}
                                title="Edit name"
                                size="small"
                            />
                        </Input.Group>
                    </div>

                    <div>
                        <Text strong style={{ display: 'block', marginBottom: '4px' }}>Diffusion Samples</Text>
                        <Input
                            value={props.foldDiffusionSamples || ''}
                            disabled
                            size="small"
                        />
                    </div>

                    {/* Row 2 */}
                    <div>
                        <Text strong style={{ display: 'block', marginBottom: '4px' }}>Owner</Text>
                        <Input
                            value={props.foldOwner}
                            disabled
                            size="small"
                        />
                    </div>

                    <div>
                        <Text strong style={{ display: 'block', marginBottom: '4px' }}>Created</Text>
                        <Input
                            value={props.foldCreateDate}
                            disabled
                            size="small"
                        />
                    </div>

                    {/* Row 3 */}
                    <div>
                        <Text strong style={{ display: 'block', marginBottom: '4px' }}>Public</Text>
                        <Switch
                            checked={props.foldPublic || false}
                            onChange={(checked) => props.setPublic(checked)}
                            checkedChildren="Public"
                            unCheckedChildren="Private"
                            size="small"
                        />
                    </div>

                    <div>
                        <Text strong style={{ display: 'block', marginBottom: '4px' }}>Model Preset</Text>
                        <Input.Group compact>
                            <Input
                                value={props.foldModelPreset || "unset"}
                                disabled
                                style={{ width: 'calc(100% - 40px)' }}
                                size="small"
                            />
                            <AntButton
                                icon={<EditOutlined />}
                                onClick={props.setFoldModelPreset}
                                title="Edit model preset"
                                size="small"
                            />
                        </Input.Group>
                    </div>

                    {/* Row 4 - spans both columns */}
                    <div style={{ gridColumn: '1 / -1' }}>
                        <Text strong style={{ display: 'block', marginBottom: '4px' }}>Disable Relaxation</Text>
                        <Switch
                            checked={props.foldDisableRelaxation !== null ? props.foldDisableRelaxation : true}
                            onChange={(checked) => props.setDisableRelaxation(checked)}
                            checkedChildren="Disabled"
                            unCheckedChildren="Enabled"
                            size="small"
                        />
                    </div>

                    {/* Tags section - spans both columns */}
                    <div style={{ gridColumn: '1 / -1' }}>
                        <Text strong style={{ display: 'block', marginBottom: '4px' }}>Tags</Text>
                        <div style={{ marginBottom: '8px' }}>
                            {(props.foldTags || []).map(tag => (
                                <Tag
                                    key={tag}
                                    closable={props.userType !== "viewer"}
                                    onClose={() => props.deleteTag(tag)}
                                    onClick={() => props.handleTagClick(tag)}
                                    style={{ cursor: 'pointer', marginBottom: '4px' }}
                                    size="small"
                                >
                                    {tag}
                                </Tag>
                            ))}
                        </div>
                        <EditableTagList
                            tags={props.foldTags || []}
                            addTag={props.addTag}
                            deleteTag={props.deleteTag}
                            handleTagClick={props.handleTagClick}
                        />
                    </div>
                </div>
            </SectionCard>
        </TabContainer>
    );
});

export default SequenceTab;
