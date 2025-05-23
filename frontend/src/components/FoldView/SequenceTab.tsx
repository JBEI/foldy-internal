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
                <FormRow>
                    <FormField>
                        <label className="uk-form-label">Name</label>
                        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                            <input
                                className="uk-input"
                                value={props.foldName}
                                disabled
                                style={{ flex: 1 }}
                            />
                            <button
                                className="uk-button uk-button-default uk-button-small"
                                onClick={(e) => {
                                    e.preventDefault();
                                    props.setFoldName();
                                }}
                                disabled={props.userType === "viewer"}
                                title="Edit name"
                            >
                                <AiFillEdit />
                            </button>
                        </div>
                    </FormField>
                </FormRow>

                <FormRow>
                    <FormField>
                        <label className="uk-form-label">Diffusion Samples</label>
                        <input
                            className="uk-input"
                            value={props.foldDiffusionSamples || ''}
                            disabled
                        />
                    </FormField>
                </FormRow>

                <FormRow>
                    <FormField>
                        <label className="uk-form-label">Owner</label>
                        <input
                            className="uk-input"
                            value={props.foldOwner}
                            disabled
                        />
                    </FormField>
                </FormRow>

                <FormRow>
                    <FormField>
                        <label className="uk-form-label">Created</label>
                        <input
                            className="uk-input"
                            value={props.foldCreateDate}
                            disabled
                        />
                    </FormField>
                </FormRow>

                <FormRow>
                    <FormField>
                        <CheckboxControl
                            label="Public"
                            checked={props.foldPublic || false}
                            onChange={(checked) => props.setPublic(checked)}
                        />
                    </FormField>
                </FormRow>

                <FormRow>
                    <FormField style={{ width: '100%' }}>
                        <label className="uk-form-label">Tags</label>
                        <EditableTagList
                            tags={props.foldTags || []}
                            addTag={props.addTag}
                            deleteTag={props.deleteTag}
                            handleTagClick={props.handleTagClick}
                        />
                    </FormField>
                </FormRow>

                <FormRow>
                    <FormField>
                        <label className="uk-form-label">Model Preset</label>
                        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                            <input
                                className="uk-input"
                                value={props.foldModelPreset || "unset"}
                                disabled
                                style={{ flex: 1 }}
                            />
                            <button
                                className="uk-button uk-button-default uk-button-small"
                                onClick={(e) => {
                                    e.preventDefault();
                                    props.setFoldModelPreset();
                                }}
                                title="Edit model preset"
                            >
                                <AiFillEdit />
                            </button>
                        </div>
                    </FormField>
                </FormRow>

                <FormRow>
                    <FormField>
                        <CheckboxControl
                            label="Disable Relaxation"
                            checked={props.foldDisableRelaxation !== null ? props.foldDisableRelaxation : true}
                            onChange={(checked) => props.setDisableRelaxation(checked)}
                        />
                    </FormField>
                </FormRow>
            </SectionCard>
        </TabContainer>
    );
});

export default SequenceTab;
