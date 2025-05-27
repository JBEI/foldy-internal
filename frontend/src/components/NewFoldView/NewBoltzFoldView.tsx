import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import BoltzYamlBuilder from "../../util/boltzYamlBuilder";
import { Row, Col, Form, Input, Switch, Alert, InputNumber, Modal, Button as AntButton, Typography, Collapse, Card } from "antd";
import { QuestionCircleOutlined, BookOutlined } from '@ant-design/icons';
import { postFolds } from "../../api/foldApi";
import { FoldInput } from "../../types/types";
import UIkit from "uikit";
import { notify } from "../../services/NotificationService";

const { Text, Paragraph, Title } = Typography;
const { Panel } = Collapse;

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
    const [showHelpModal, setShowHelpModal] = useState<boolean>(false);
    const [showCitationsModal, setShowCitationsModal] = useState<boolean>(false);
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

                {/* Clean overview with help buttons */}
                <Alert
                    message="Create a new protein structure prediction"
                    description={
                        <div>
                            <Paragraph>
                                Foldy uses Boltz-1x to predict protein structures with exceptional accuracy for multimers,
                                small molecule docking, DNA/RNA interactions, and post-translational modifications.
                            </Paragraph>
                            <div style={{ display: 'flex', gap: '12px', marginTop: '12px' }}>
                                <AntButton
                                    type="link"
                                    icon={<QuestionCircleOutlined />}
                                    onClick={() => setShowHelpModal(true)}
                                    style={{ padding: 0 }}
                                >
                                    View new fold guide
                                </AntButton>
                                <AntButton
                                    type="link"
                                    icon={<BookOutlined />}
                                    onClick={() => setShowCitationsModal(true)}
                                    style={{ padding: 0 }}
                                >
                                    View citations
                                </AntButton>
                            </div>
                        </div>
                    }
                    type="info"
                    showIcon
                    style={{ marginBottom: '20px' }}
                />

                {/* Setup Instructions Modal */}
                <Modal
                    title="Boltz Fold Guide"
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
                        <Title level={4}>What is Boltz-1x?</Title>
                        <Paragraph>
                            Boltz-1x is an open-source protein structure prediction model with exceptional accuracy for complex scenarios:
                        </Paragraph>
                        <ul>
                            <li>Protein multimers (multi-chain complexes)</li>
                            <li>Small molecule docking</li>
                            <li>DNA/RNA interactions</li>
                            <li>Post-translational modifications</li>
                        </ul>
                        <Paragraph>
                            <a href="https://github.com/jwohlwend/boltz" target="_blank" rel="noopener noreferrer">GitHub Repository</a> |
                            <a href="https://www.biorxiv.org/content/10.1101/2024.11.19.624167v4" target="_blank" rel="noopener noreferrer"> Research Paper</a>
                        </Paragraph>

                        <Title level={4}>Required Inputs</Title>
                        <ul>
                            <li>
                                <Text strong>Fold Name:</Text> Unique identifier (max 80 characters, use only letters, numbers, underscores, hyphens, and spaces)
                            </li>
                            <li>
                                <Text strong>YAML Configuration:</Text> Define sequences and structure
                            </li>
                            <li>
                                <Text strong>Chain IDs:</Text> Single uppercase letters for each molecule (A, B, C, etc.)
                            </li>
                        </ul>

                        <Title level={4}>YAML Structure</Title>
                        <ul>
                            <li><Text strong>Version:</Text> Always set to 1</li>
                            <li><Text strong>Sequences:</Text> Define protein, DNA, RNA, and ligand sequences</li>
                            <li><Text strong>Multiple copies:</Text> Use multiple chain IDs for the same sequence</li>
                        </ul>

                        <Alert
                            message="Need help with YAML?"
                            description="Use the interactive YAML builder below to construct your configuration step-by-step."
                            type="success"
                            showIcon
                            style={{ marginTop: '16px' }}
                        />
                    </div>
                </Modal>

                {/* Citations Modal */}
                <Modal
                    title="Citations & References"
                    open={showCitationsModal}
                    onCancel={() => setShowCitationsModal(false)}
                    footer={[
                        <AntButton key="close" onClick={() => setShowCitationsModal(false)}>
                            Close
                        </AntButton>
                    ]}
                    width={800}
                >
                    <div>
                        <Paragraph>
                            If you use results from this platform in your research, please cite the appropriate papers:
                        </Paragraph>

                        <Collapse ghost>

                            <Panel header={<Text strong>🧪 FolDE (If you used FolDE for structure prediction or directed evolution)</Text>} key="2">
                                <Card size="small" style={{ backgroundColor: '#f9f9f9' }}>
                                    <Text>Include this citation if you use the FolDE website, for structure prediction or directed evolution, in your research:</Text>
                                    <Alert
                                        message="Not Yet Published"
                                        description="FolDE manuscript is currently in preparation. Please check back for citation details."
                                        type="info"
                                        showIcon
                                        size="small"
                                        style={{ marginTop: '8px' }}
                                    />
                                </Card>
                            </Panel>

                            <Panel header={<Text strong>📝 Boltz-1: Primary Citation (Required)</Text>} key="1">
                                <Card size="small" style={{ backgroundColor: '#f9f9f9' }}>
                                    <Text>Use this citation if you use Boltz-1 or Boltz-1x structure predictions:</Text>
                                    <pre style={{
                                        marginTop: '8px',
                                        padding: '12px',
                                        backgroundColor: '#fff',
                                        border: '1px solid #d9d9d9',
                                        borderRadius: '4px',
                                        fontSize: '12px',
                                        lineHeight: '1.4'
                                    }}>
                                        {`@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and
            Getz, Noah and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and
            Atkinson, Liam and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and
            Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}`}
                                    </pre>
                                </Card>
                            </Panel>

                            <Panel header={<Text strong>🧬 ColabFold: MSA Generation (If Applicable)</Text>} key="3">
                                <Card size="small" style={{ backgroundColor: '#f9f9f9' }}>
                                    <Text>Include this citation if Foldy generated Multiple Sequence Alignments (MSAs) for your fold:</Text>
                                    <pre style={{
                                        marginTop: '8px',
                                        padding: '12px',
                                        backgroundColor: '#fff',
                                        border: '1px solid #d9d9d9',
                                        borderRadius: '4px',
                                        fontSize: '12px',
                                        lineHeight: '1.4'
                                    }}>
                                        {`@article{mirdita2022colabfold,
  title = {ColabFold: making protein folding accessible to all},
  author = {Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and
            Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal = {Nature Methods},
  volume = {19},
  number = {6},
  pages = {679--682},
  year = {2022},
  publisher = {Nature Publishing Group}
}`}
                                    </pre>
                                </Card>
                            </Panel>
                        </Collapse>

                        <Alert
                            message="Pro Tip"
                            description="Most reference managers can import these citations directly from DOI or journal information."
                            type="info"
                            showIcon
                            style={{ marginTop: '16px' }}
                        />
                    </div>
                </Modal>
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
