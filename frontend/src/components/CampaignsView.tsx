import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Table, Button, Modal, Form, Input, Select, Typography, Space, Card, Pagination, Tag } from 'antd';
import { PlusOutlined, EyeOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons';
import { ColumnsType } from 'antd/es/table';
import { Campaign } from '../types/types';
import { PaginatedCampaignsResponse, getCampaigns, createCampaign, deleteCampaign } from '../api/campaignApi';
import { getFoldsWithPagination } from '../api/foldApi';
import { notify } from '../services/NotificationService';
import { ESMModelPicker } from './FoldView/ESMModelPicker';

const { Title, Text } = Typography;
const { TextArea } = Input;

interface Fold {
    id: number;
    name: string;
}

const CampaignsView: React.FC = () => {
    const navigate = useNavigate();
    const [campaigns, setCampaigns] = useState<Campaign[]>([]);
    const [loading, setLoading] = useState(false);
    const [totalCampaigns, setTotalCampaigns] = useState(0);
    const [currentPage, setCurrentPage] = useState(1);
    const [pageSize] = useState(20);

    const [folds, setFolds] = useState<Fold[]>([]);
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [createForm] = Form.useForm();

    const loadCampaigns = async (page: number = 1) => {
        setLoading(true);
        try {
            const response: PaginatedCampaignsResponse = await getCampaigns(page, pageSize);
            setCampaigns(response.campaigns);
            setTotalCampaigns(response.total);
            setCurrentPage(response.page);
        } catch (error) {
            notify.error('Failed to load campaigns');
            console.error('Error loading campaigns:', error);
        } finally {
            setLoading(false);
        }
    };

    const loadFolds = async () => {
        try {
            const foldsResponse = await getFoldsWithPagination(null, null, 1, 1000);
            setFolds(foldsResponse.data.map((fold: any) => ({ id: fold.id, name: fold.name })));
        } catch (error) {
            notify.error('Failed to load folds');
            console.error('Error loading folds:', error);
        }
    };

    useEffect(() => {
        loadCampaigns();
        loadFolds();
    }, []);

    const handleCreateCampaign = async (values: any) => {
        try {
            await createCampaign({
                name: values.name,
                fold_id: values.fold_id,
                description: values.description,
                naturalness_model: values.naturalness_model,
                embedding_model: values.embedding_model,
            });
            notify.success('Campaign created successfully');
            setShowCreateModal(false);
            createForm.resetFields();
            loadCampaigns(currentPage);
        } catch (error: any) {
            notify.error(error.response?.data?.message || 'Failed to create campaign');
        }
    };

    const handleDeleteCampaign = async (campaignId: number, campaignName: string) => {
        Modal.confirm({
            title: 'Delete Campaign',
            content: `Are you sure you want to delete campaign "${campaignName}"? This action cannot be undone.`,
            okText: 'Delete',
            okType: 'danger',
            onOk: async () => {
                try {
                    await deleteCampaign(campaignId);
                    notify.success('Campaign deleted successfully');
                    loadCampaigns(currentPage);
                } catch (error: any) {
                    notify.error(error.response?.data?.message || 'Failed to delete campaign');
                }
            },
        });
    };

    const columns: ColumnsType<Campaign> = [
        {
            title: 'Name',
            dataIndex: 'name',
            key: 'name',
            render: (name: string, record: Campaign) => (
                <Button
                    type="link"
                    onClick={() => navigate(`/campaigns/${record.id}`)}
                    style={{ padding: 0, height: 'auto' }}
                >
                    {name}
                </Button>
            ),
        },
        {
            title: 'Fold',
            dataIndex: 'fold_name',
            key: 'fold_name',
            render: (foldName: string, record: Campaign) => (
                <Tag color="blue">
                    <Button
                        type="link"
                        onClick={() => navigate(`/fold/${record.fold_id}`)}
                        style={{
                            padding: 0,
                            height: 'auto',
                            color: 'inherit',
                            fontSize: 'inherit'
                        }}
                    >
                        {foldName}
                    </Button>
                </Tag>
            ),
        },
        {
            title: 'Description',
            dataIndex: 'description',
            key: 'description',
            ellipsis: true,
            render: (description: string) => description || <Text type="secondary">No description</Text>,
        },
        {
            title: 'Rounds',
            key: 'rounds',
            render: (_, record: Campaign) => (
                <Text>{record.rounds?.length || 0} rounds</Text>
            ),
        },
        {
            title: 'Created',
            dataIndex: 'created_at',
            key: 'created_at',
            render: (date: string) => new Date(date).toLocaleDateString(),
        },
        {
            title: 'Actions',
            key: 'actions',
            width: 120,
            render: (_, record: Campaign) => (
                <Space>
                    <Button
                        type="text"
                        icon={<EyeOutlined />}
                        onClick={() => navigate(`/campaigns/${record.id}`)}
                        title="View campaign"
                    />
                    <Button
                        type="text"
                        danger
                        icon={<DeleteOutlined />}
                        onClick={() => handleDeleteCampaign(record.id, record.name)}
                        title="Delete campaign"
                    />
                </Space>
            ),
        },
    ];

    return (
        <div style={{ padding: '24px' }}>
            <Card>
                <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                        <Title level={2} style={{ margin: 0 }}>Campaigns (In Construction)</Title>
                        <Text type="secondary">Manage your directed evolution campaigns</Text>
                    </div>
                    <Button
                        type="primary"
                        icon={<PlusOutlined />}
                        onClick={() => setShowCreateModal(true)}
                    >
                        New Campaign
                    </Button>
                </div>

                <Table
                    columns={columns}
                    dataSource={campaigns}
                    loading={loading}
                    rowKey="id"
                    pagination={false}
                    style={{ marginBottom: '16px' }}
                />

                <div style={{ textAlign: 'right' }}>
                    <Pagination
                        current={currentPage}
                        total={totalCampaigns}
                        pageSize={pageSize}
                        onChange={(page) => {
                            setCurrentPage(page);
                            loadCampaigns(page);
                        }}
                        showSizeChanger={false}
                        showQuickJumper
                        showTotal={(total, range) =>
                            `${range[0]}-${range[1]} of ${total} campaigns`
                        }
                    />
                </div>
            </Card>

            <Modal
                title="Create New Campaign"
                open={showCreateModal}
                onCancel={() => {
                    setShowCreateModal(false);
                    createForm.resetFields();
                }}
                footer={null}
            >
                <Form
                    form={createForm}
                    layout="vertical"
                    onFinish={handleCreateCampaign}
                >
                    <Form.Item
                        name="name"
                        label="Campaign Name"
                        rules={[{ required: true, message: 'Please enter a campaign name' }]}
                    >
                        <Input placeholder="e.g., High Activity Evolution" />
                    </Form.Item>

                    <Form.Item
                        name="fold_id"
                        label="Fold"
                        rules={[{ required: true, message: 'Please select a fold' }]}
                    >
                        <Select
                            placeholder="Select a fold for this campaign"
                            showSearch
                            optionFilterProp="children"
                        >
                            {folds.map(fold => (
                                <Select.Option key={fold.id} value={fold.id}>
                                    {fold.name}
                                </Select.Option>
                            ))}
                        </Select>
                    </Form.Item>

                    <Form.Item
                        name="description"
                        label="Description (Optional)"
                    >
                        <TextArea
                            rows={3}
                            placeholder="Describe the goals and approach for this campaign..."
                        />
                    </Form.Item>

                    <Form.Item
                        name="naturalness_model"
                        label="Naturalness Protein Language Model"
                        initialValue="esm2_t33_650M_UR50D"
                    >
                        <ESMModelPicker
                            value={createForm.getFieldValue('naturalness_model') || "esm2_t33_650M_UR50D"}
                            onChange={(value) => createForm.setFieldValue('naturalness_model', value)}
                            label=""
                        />
                    </Form.Item>

                    <Form.Item
                        name="embedding_model"
                        label="Embedding Model"
                        initialValue="esm2_t33_650M_UR50D"
                    >
                        <ESMModelPicker
                            value={createForm.getFieldValue('embedding_model') || "esm2_t33_650M_UR50D"}
                            onChange={(value) => createForm.setFieldValue('embedding_model', value)}
                            label=""
                        />
                    </Form.Item>

                    <Form.Item style={{ marginTop: '24px', marginBottom: 0 }}>
                        <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
                            <Button onClick={() => {
                                setShowCreateModal(false);
                                createForm.resetFields();
                            }}>
                                Cancel
                            </Button>
                            <Button type="primary" htmlType="submit">
                                Create Campaign
                            </Button>
                        </Space>
                    </Form.Item>
                </Form>
            </Modal>
        </div>
    );
};

export default CampaignsView;
