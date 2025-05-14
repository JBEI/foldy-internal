import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import BoltzYamlBuilder from "../../util/boltzYamlBuilder";
import { Row, Col, Form, Input, Switch, Alert, InputNumber } from "antd";
import { postFolds } from "../../api/foldApi";
import { FoldInput } from "../../types/types";
import UIkit from "uikit";
import { notify } from "../../services/NotificationService";

interface NewBoltzFoldViewProps {
    userType: string | null;
}

interface AdvancedSettings {
    diffusionSamples: number;
    startFoldJob: boolean;
    emailOnCompletion: boolean;
    skipDuplicateEntries: boolean;
    stayOnPage: boolean;
}

async function createFold(
    foldName: string,
    yamlData: string,
    options: {
        userType: string | null;
        diffusionSamples?: number;
        startFoldJob?: boolean;
        emailOnCompletion?: boolean;
        skipDuplicateEntries?: boolean;
    }
): Promise<void> {
    if (options.userType === "viewer") {
        throw new Error("Viewers cannot create folds");
    }

    const fold: FoldInput = {
        name: foldName,
        tags: [],
        yaml_config: yamlData,
        diffusion_samples: options.diffusionSamples || null,
        yaml_helper: null,
        sequence: null,
        af2_model_preset: "boltz",
        disable_relaxation: false,
    };

    return await postFolds([fold], {
        startJob: options.startFoldJob || false,
        emailOnCompletion: options.emailOnCompletion || false,
        skipDuplicates: options.skipDuplicateEntries || false,
    });
}

const NewBoltzFoldView: React.FC<NewBoltzFoldViewProps> = ({ userType }) => {
    const navigate = useNavigate();
    const [foldName, setFoldName] = useState("");
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [advancedSettings, setAdvancedSettings] = useState<AdvancedSettings>({
        diffusionSamples: 1,
        startFoldJob: true,
        emailOnCompletion: true,
        skipDuplicateEntries: false,
        stayOnPage: false,
    });

    // Example partial YAML (you can keep or remove this)
    const partialYaml = `
version: 1
sequences:
`;

    async function handleSave(yamlString: string) {
        if (!foldName.trim()) {
            notify.warning("Please enter a fold name");
            return;
        }

        // Check for weird characters in the fold name
        if (/[^a-zA-Z0-9_ -]/.test(foldName)) {
            notify.warning("Fold name contains invalid characters. Please use only letters, numbers, underscores, hyphens, and spaces.");
            return;
        }

        setIsSubmitting(true);
        try {
            await createFold(foldName, yamlString, {
                userType,
                ...advancedSettings,
            });

            notify.success("Fold successfully created!");

            if (!advancedSettings.stayOnPage) {
                navigate("/");
            }
        } catch (err) {
            console.error(err);
            notify.error(`Failed to create fold: ${String(err)}`);
        } finally {
            setIsSubmitting(false);
        }
    }

    return (
        <div
            data-testid="About"
            style={{
                flexGrow: 1,
                overflowY: "scroll",
                paddingTop: "10px",
                paddingBottom: "10px",
            }}>
            {/* Fixed Header */}
            <div style={{ padding: "1rem", borderBottom: "1px solid #f0f0f0" }}>
                <h1>New Boltz Fold</h1>

                {userType === "viewer" && (
                    <Alert
                        message="You do not have permissions to submit folds on this instance."
                        type="error"
                        style={{ marginBottom: "1rem" }}
                    />
                )}

                <div>
                    <p>
                        Foldy is built on Boltz-1x for protein structure prediction (<a href="https://github.com/jwohlwend/boltz">Github</a>, <a href="https://www.biorxiv.org/content/10.1101/2024.11.19.624167v4">Paper</a>). Boltz-1x is an open-source model for predicting protein structures and has exceptional accuracy for many problem types including:
                    </p>
                    <ul style={{ marginLeft: "2rem", marginBottom: "1rem" }}>
                        <li>Protein multimers</li>
                        <li>Small molecule docking</li>
                        <li>DNA/RNA docking</li>
                        <li>Post translational modifications</li>
                    </ul>
                    <p>To predict a structure, supply:</p>
                    <ul style={{ marginLeft: "2rem", marginBottom: "1rem" }}>
                        <li>Fold Name: your fold name should be unique, and we recommend choosing something less than 80 characters and only using [0-9a-zA-Z_\- ]</li>
                        <li>YAML version: There is only one version number, just leave as 1</li>
                        <li>Chain IDs: Each molecule (protein, ligand, DNA, RNA) requires a chain ID, and a common convention is single upper case characters. If you want two copies of a molecule you can provide multiple chain names.</li>
                    </ul>
                    <p>If you use Boltz-1 or Boltz-1x structures in your work, please cite the Boltz paper:</p>
                    <pre style={{ marginBottom: "1rem" }}>{`@article{wohlwend2024boltz1,
    author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Getz, Noah and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Atkinson, Liam and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
    title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
    year = {2024},
    doi = {10.1101/2024.11.19.624167},
    journal = {bioRxiv}
}`}
                    </pre>
                    <p>If you use Foldy to run Boltz with automatic Multiple Sequence Alignment (MSA) generation, please also cite:</p>
                    <pre style={{ marginBottom: "1rem" }}>{`@article{mirdita2022colabfold,
    title={ColabFold: making protein folding accessible to all},
    author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
    journal={Nature methods},
    year={2022},
}`}
                    </pre>
                </div>
            </div>

            {/* Scrollable Content */}
            <div style={{ flex: 1, overflow: "auto", padding: "1rem" }}>
                <Row gutter={24}>
                    <Col span={18}>
                        <Form.Item
                            label="Fold Name"
                            required
                            style={{ marginBottom: "2rem" }}
                        >
                            <Input
                                value={foldName}
                                onChange={(e) => setFoldName(e.target.value)}
                                placeholder="Enter fold name"
                                disabled={userType === "viewer"}
                            />
                        </Form.Item>

                        <BoltzYamlBuilder
                            initialYaml={partialYaml}
                            onSave={handleSave}
                        />
                    </Col>

                    {/* Advanced settings column - will scroll with content */}
                    <Col span={6} style={{ position: "sticky", top: "1rem" }}>
                        <div style={{
                            backgroundColor: "#f5f5f5",
                            padding: "1rem",
                            borderRadius: "8px"
                        }}>
                            <h3>Advanced Settings</h3>
                            <Form layout="vertical">
                                <Form.Item label="Diffusion Samples">
                                    <InputNumber
                                        value={advancedSettings.diffusionSamples}
                                        onChange={(value) =>
                                            setAdvancedSettings((prev) => ({
                                                ...prev,
                                                diffusionSamples: value || 1,
                                            }))
                                        }
                                        disabled={userType === "viewer"}
                                    />
                                </Form.Item>


                                <Form.Item label="Start Fold Job Immediately">
                                    <Switch
                                        checked={advancedSettings.startFoldJob}
                                        onChange={(checked) =>
                                            setAdvancedSettings((prev) => ({
                                                ...prev,
                                                startFoldJob: checked,
                                            }))
                                        }
                                        disabled={userType === "viewer"}
                                    />
                                </Form.Item>

                                <Form.Item label="Email on Completion">
                                    <Switch
                                        checked={advancedSettings.emailOnCompletion}
                                        onChange={(checked) =>
                                            setAdvancedSettings((prev) => ({
                                                ...prev,
                                                emailOnCompletion: checked,
                                            }))
                                        }
                                        disabled={userType === "viewer"}
                                    />
                                </Form.Item>

                                <Form.Item label="Skip Duplicate Entries">
                                    <Switch
                                        checked={advancedSettings.skipDuplicateEntries}
                                        onChange={(checked) =>
                                            setAdvancedSettings((prev) => ({
                                                ...prev,
                                                skipDuplicateEntries: checked,
                                            }))
                                        }
                                        disabled={userType === "viewer"}
                                    />
                                </Form.Item>

                                <Form.Item label="Stay on this page after fold creation">
                                    <Switch
                                        checked={advancedSettings.stayOnPage}
                                        onChange={(checked) =>
                                            setAdvancedSettings((prev) => ({
                                                ...prev,
                                                stayOnPage: checked,
                                            }))
                                        }
                                        disabled={userType === "viewer"}
                                    />
                                </Form.Item>
                            </Form>
                        </div>
                    </Col>
                </Row>
            </div>
        </div>
    );
};

export default NewBoltzFoldView;
