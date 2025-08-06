import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, useLocation, Link } from 'react-router-dom';
import {
    Layout,
    Card,
    Typography,
    Steps,
    Button,
    Space,
    Modal,
    Form,
    Input,
    Descriptions,
    Spin,
    Empty,
    Row,
    Col,
} from 'antd';
import {
    ArrowLeftOutlined,
    PlusOutlined,
    EditOutlined,
    DeleteOutlined,
    CalendarOutlined,
    HomeOutlined,
    ExperimentOutlined
} from '@ant-design/icons';
import { Campaign, CampaignRound } from '../types/types';
import {
    getCampaign,
    updateCampaign,
    createCampaignRound,
    deleteCampaignRound,
    getCampaignRound
} from '../api/campaignApi';
import { notify } from '../services/NotificationService';
import { ESMModelPicker } from './FoldView/ESMModelPicker';
import CampaignRoundComponent from './CampaignView/CampaignRound';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { Sider, Content } = Layout;

const CampaignView: React.FC = () => {
    const { campaignId, roundNumber } = useParams<{ campaignId: string; roundNumber?: string }>();
    const navigate = useNavigate();
    const location = useLocation();

    const [campaign, setCampaign] = useState<Campaign | null>(null);
    const [loading, setLoading] = useState(true);
    const [showEditModal, setShowEditModal] = useState(false);
    const [showNewRoundModal, setShowNewRoundModal] = useState(false);
    const [editForm] = Form.useForm();
    const [roundForm] = Form.useForm();

    // Responsive layout state
    const [isMobile, setIsMobile] = useState(false);

    // Check screen size for responsive layout
    useEffect(() => {
        const checkScreenSize = () => {
            setIsMobile(window.innerWidth < 992); // Bootstrap lg breakpoint
        };

        checkScreenSize();
        window.addEventListener('resize', checkScreenSize);

        return () => window.removeEventListener('resize', checkScreenSize);
    }, []);

    // Determine if we're on overview or a specific round
    const isOverview = !roundNumber;
    const currentRoundNumber = roundNumber ? parseInt(roundNumber) : null;

    const loadCampaign = async () => {
        if (!campaignId) return;

        setLoading(true);
        try {
            const campaignData = await getCampaign(parseInt(campaignId));
            setCampaign(campaignData);
        } catch (error) {
            notify.error('Failed to load campaign');
            console.error('Error loading campaign:', error);
            navigate('/campaigns');
        } finally {
            setLoading(false);
        }
    };

    const refreshCurrentRound = async () => {
        if (!campaignId || !currentRoundNumber || !campaign) return;

        try {
            const updatedRound = await getCampaignRound(parseInt(campaignId), currentRoundNumber);

            // Update only the specific round in the campaign
            setCampaign(prevCampaign => {
                if (!prevCampaign) return prevCampaign;

                const updatedRounds = prevCampaign.rounds?.map(round =>
                    round.round_number === currentRoundNumber ? updatedRound : round
                ) || [];

                return {
                    ...prevCampaign,
                    rounds: updatedRounds
                };
            });
        } catch (error) {
            console.error('Error refreshing round:', error);
            // Fallback to full campaign refresh if the targeted refresh fails
            loadCampaign();
        }
    };

    useEffect(() => {
        loadCampaign();
    }, [campaignId]);

    const handleUpdateCampaign = async (values: any) => {
        if (!campaign) return;

        // Check if configuration fields are being changed
        const isConfigChange =
            values.naturalness_model !== campaign.naturalness_model ||
            values.embedding_model !== campaign.embedding_model ||
            values.domain_boundaries !== campaign.domain_boundaries;

        const performUpdate = async () => {
            try {
                const updatedCampaign = await updateCampaign(campaign.id, {
                    name: values.name,
                    description: values.description,
                    naturalness_model: values.naturalness_model,
                    embedding_model: values.embedding_model,
                    domain_boundaries: values.domain_boundaries,
                });
                setCampaign(updatedCampaign);
                notify.success('Campaign updated successfully');
                setShowEditModal(false);
            } catch (error: any) {
                notify.error(error.response?.data?.message || 'Failed to update campaign');
            }
        };

        if (isConfigChange && sortedRounds.length > 0) {
            Modal.confirm({
                title: 'WARNING: Change Campaign Configuration?',
                content: 'Are you sure you want to change campaign configuration? This might lead to incompatibilities with earlier rounds.',
                okText: 'Yes, Update Configuration',
                okType: 'danger',
                cancelText: 'Cancel',
                onOk: performUpdate,
            });
        } else {
            performUpdate();
        }
    };

    const handleCreateRound = async (values: any) => {
        if (!campaign) return;

        try {
            const newRound = await createCampaignRound(campaign.id, {
                round_number: undefined, // Always auto-increment
            });

            // Reload the campaign to get updated rounds
            await loadCampaign();
            notify.success(`Round ${newRound.round_number} created successfully`);
            setShowNewRoundModal(false);
            roundForm.resetFields();

            // Auto-navigate to the newly created round
            navigate(`/campaigns/${campaignId}/${newRound.round_number}`);
        } catch (error: any) {
            notify.error(error.response?.data?.message || 'Failed to create round');
        }
    };

    if (loading) {
        return (
            <div style={{ padding: '24px', textAlign: 'center' }}>
                <Spin size="large" />
            </div>
        );
    }

    if (!campaign) {
        return (
            <div style={{ padding: '24px' }}>
                <Empty description="Campaign not found" />
            </div>
        );
    }

    const sortedRounds = campaign.rounds
        ? [...campaign.rounds].sort((a, b) => a.round_number - b.round_number)
        : [];

    // Create steps for navigation
    const stepItems = [
        {
            title: 'Overview',
            icon: <HomeOutlined />,
            onClick: () => navigate(`/campaigns/${campaignId}`)
        },
        ...sortedRounds.map(round => ({
            title: `Round ${round.round_number}`,
            icon: <ExperimentOutlined />,
            description: new Date(round.date_started).toLocaleDateString(),
            onClick: () => navigate(`/campaigns/${campaignId}/${round.round_number}`)
        }))
    ];

    const currentStepIndex = isOverview ? 0 : (currentRoundNumber ? sortedRounds.findIndex(r => r.round_number === currentRoundNumber) + 1 : 0);
    const currentRound = currentRoundNumber ? sortedRounds.find(r => r.round_number === currentRoundNumber) : null;

    // Render steps navigation component
    const renderStepsNavigation = () => (
        <div>
            <div style={{ marginBottom: '16px' }}>
                <Title level={4} style={{ margin: 0, fontSize: '16px' }}>
                    {campaign.name}
                </Title>
                <Link
                    to={`/fold/${campaign.fold_id}`}
                    style={{
                        fontSize: '12px',
                        color: '#8c8c8c',
                        textDecoration: 'none'
                    }}
                >
                    {campaign.fold_name}
                </Link>
            </div>

            <Steps
                direction={isMobile ? "horizontal" : "vertical"}
                current={currentStepIndex}
                items={stepItems.map((item, index) => ({
                    ...item,
                    status: index === currentStepIndex ? 'process' : (index < currentStepIndex ? 'finish' : 'wait'),
                    style: { cursor: 'pointer' }
                }))}
                onChange={(current) => {
                    if (stepItems[current]?.onClick) {
                        stepItems[current].onClick();
                    }
                }}
                style={{
                    background: 'transparent',
                    ...(isMobile && { marginBottom: '16px' })
                }}
            />

            {/* Show New Round button only if no rounds exist yet */}
            {sortedRounds.length === 0 && (
                <div style={{
                    marginTop: isMobile ? '16px' : '24px',
                    paddingTop: '16px',
                    borderTop: '1px solid #f0f0f0'
                }}>
                    <Button
                        type="primary"
                        icon={<PlusOutlined />}
                        onClick={() => setShowNewRoundModal(true)}
                        block={!isMobile}
                        size="small"
                    >
                        Start First Round
                    </Button>
                </div>
            )}
        </div>
    );

    const renderOverview = () => (
        <div>
            <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                    <Title level={2} style={{ margin: 0 }}>
                        {campaign.name}
                    </Title>
                    <Text type="secondary">
                        Campaign for <Link
                            to={`/fold/${campaign.fold_id}`}
                            style={{
                                color: 'inherit',
                                textDecoration: 'none'
                            }}
                        >
                            {campaign.fold_name}
                        </Link>
                    </Text>
                    {campaign.description && (
                        <p style={{ margin: '8px 0 0 0', color: '#666' }}>
                            {campaign.description}
                        </p>
                    )}
                </div>
                <Button
                    icon={<EditOutlined />}
                    onClick={() => {
                        editForm.setFieldsValue({
                            name: campaign.name,
                            description: campaign.description || '',
                            naturalness_model: campaign.naturalness_model || 'esm2_t33_650M_UR50D',
                            embedding_model: campaign.embedding_model || 'esm2_t33_650M_UR50D',
                            domain_boundaries: campaign.domain_boundaries || '',
                        });
                        setShowEditModal(true);
                    }}
                >
                    Edit Campaign
                </Button>
            </div>

            <Row gutter={[24, 24]}>
                <Col span={24}>
                    <Card title="Configuration">
                        <Descriptions column={2} bordered>
                            <Descriptions.Item label="Created">
                                {new Date(campaign.created_at).toLocaleString()}
                            </Descriptions.Item>
                            <Descriptions.Item label="Naturalness Model">
                                <Text code>{campaign.naturalness_model || 'esm2_t33_650M_UR50D'}</Text>
                            </Descriptions.Item>
                            <Descriptions.Item label="Embedding Model">
                                <Text code>{campaign.embedding_model || 'esm2_t33_650M_UR50D'}</Text>
                            </Descriptions.Item>
                            <Descriptions.Item label="Domain Boundaries">
                                {campaign.domain_boundaries ? (
                                    <Text code>{campaign.domain_boundaries}</Text>
                                ) : (
                                    <Text type="secondary">Not set</Text>
                                )}
                            </Descriptions.Item>
                        </Descriptions>
                    </Card>
                </Col>

            </Row>
        </div>
    );

    const renderRound = () => {
        if (!currentRound) {
            return <Empty description="Round not found" />;
        }

        // For now, all rounds use the new workflow
        // Later we can add logic here to determine if a round is "finished"
        // and show a different view for completed rounds
        return (
            <CampaignRoundComponent
                campaign={campaign}
                currentRound={currentRound}
                onRefresh={loadCampaign}
                onRefreshRound={refreshCurrentRound}
                onStartNewRound={() => setShowNewRoundModal(true)}
            />
        );
    };

    return (
        <div style={{ padding: '24px', minHeight: 'calc(100vh - 64px)' }}>
            <div style={{ marginBottom: '24px' }}>
                <Button
                    icon={<ArrowLeftOutlined />}
                    onClick={() => navigate('/campaigns')}
                >
                    Back to Campaigns
                </Button>
            </div>

            {isMobile ? (
                // Mobile Layout: Stacked vertically
                <div style={{ background: '#fff', minHeight: '600px', display: 'flex', flexDirection: 'column', height: 'calc(100vh - 112px)' }}>
                    {/* Steps navigation on top for mobile */}
                    <div style={{
                        background: '#fafafa',
                        borderBottom: '1px solid #f0f0f0',
                        padding: '16px',
                        maxHeight: '300px',
                        overflowY: 'auto'
                    }}>
                        {renderStepsNavigation()}
                    </div>

                    {/* Main content below, scrollable */}
                    <div style={{
                        padding: '24px',
                        flex: 1,
                        overflowY: 'auto'
                    }}>
                        {isOverview ? renderOverview() : renderRound()}
                    </div>
                </div>
            ) : (
                // Desktop Layout: Side-by-side
                <Layout style={{ background: '#fff', minHeight: '600px' }}>
                    <Sider
                        width={280}
                        style={{
                            background: '#fafafa',
                            borderRight: '1px solid #f0f0f0',
                            padding: '16px'
                        }}
                    >
                        {renderStepsNavigation()}
                    </Sider>

                    <Content style={{
                        padding: '24px',
                        maxHeight: 'calc(100vh - 112px)',
                        overflowY: 'auto',
                        flex: 1
                    }}>
                        {isOverview ? renderOverview() : renderRound()}
                    </Content>
                </Layout>
            )}

            {/* Edit Campaign Modal */}
            <Modal
                title="Edit Campaign"
                open={showEditModal}
                onCancel={() => setShowEditModal(false)}
                footer={null}
            >
                <Form
                    form={editForm}
                    layout="vertical"
                    onFinish={handleUpdateCampaign}
                >
                    <Form.Item
                        name="name"
                        label="Campaign Name"
                        rules={[{ required: true, message: 'Please enter a campaign name' }]}
                    >
                        <Input />
                    </Form.Item>

                    <Form.Item
                        name="description"
                        label="Description"
                    >
                        <TextArea rows={4} />
                    </Form.Item>

                    <Form.Item
                        name="naturalness_model"
                        label="Naturalness Protein Language Model"
                    >
                        <ESMModelPicker
                            value={editForm.getFieldValue('naturalness_model') || "esm2_t33_650M_UR50D"}
                            onChange={(value) => editForm.setFieldValue('naturalness_model', value)}
                            label=""
                        />
                    </Form.Item>

                    <Form.Item
                        name="embedding_model"
                        label="Embedding Model"
                    >
                        <ESMModelPicker
                            value={editForm.getFieldValue('embedding_model') || "esm2_t33_650M_UR50D"}
                            onChange={(value) => editForm.setFieldValue('embedding_model', value)}
                            label=""
                        />
                    </Form.Item>

                    <Form.Item
                        name="domain_boundaries"
                        label="Domain Boundaries"
                        tooltip="Optional comma-separated list of boundaries for domain-pooling when generating embeddings (e.g., 10,50,100)"
                    >
                        <Input placeholder="e.g., 10,50,100" />
                    </Form.Item>

                    <Form.Item style={{ marginBottom: 0 }}>
                        <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
                            <Button onClick={() => setShowEditModal(false)}>
                                Cancel
                            </Button>
                            <Button type="primary" htmlType="submit">
                                Update Campaign
                            </Button>
                        </Space>
                    </Form.Item>
                </Form>
            </Modal>

            {/* New Round Modal */}
            <Modal
                title="Create New Round"
                open={showNewRoundModal}
                onCancel={() => {
                    setShowNewRoundModal(false);
                    roundForm.resetFields();
                }}
                footer={null}
            >
                <Form
                    form={roundForm}
                    layout="vertical"
                    onFinish={handleCreateRound}
                >
                    <div style={{ marginBottom: '16px', textAlign: 'center' }}>
                        <p>This will create Round {sortedRounds.length + 1} for your campaign.</p>
                    </div>

                    <Form.Item style={{ marginBottom: 0 }}>
                        <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
                            <Button onClick={() => {
                                setShowNewRoundModal(false);
                                roundForm.resetFields();
                            }}>
                                Cancel
                            </Button>
                            <Button type="primary" htmlType="submit">
                                Create Round
                            </Button>
                        </Space>
                    </Form.Item>
                </Form>
            </Modal>
        </div>
    );
};

export default CampaignView;
