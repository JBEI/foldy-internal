import React from 'react';
import { Typography, Input, Space } from 'antd';

const { Title } = Typography;

interface PageHeaderProps {
    title: string;
    searchValue?: string;
    searchPlaceholder?: string;
    onSearchChange?: (value: string) => void;
    showSearch?: boolean;
    actions?: React.ReactNode;
}

export const PageHeader: React.FC<PageHeaderProps> = ({
    title,
    searchValue,
    searchPlaceholder = "Search...",
    onSearchChange,
    showSearch = true,
    actions
}) => {
    return (
        <div style={{ marginBottom: '24px' }}>
            {/* Desktop Layout */}
            <div style={{
                display: 'flex',
                gap: '16px',
                alignItems: 'center',
                flexWrap: 'wrap'
            }}>
                {/* Title - Always visible */}
                <Title
                    level={2}
                    style={{
                        margin: 0,
                        minWidth: 'fit-content'
                    }}
                >
                    {title}
                </Title>

                {/* Search Bar - Grows to fill space on larger screens */}
                {showSearch && (
                    <Input.Search
                        placeholder={searchPlaceholder}
                        value={searchValue}
                        onChange={(e) => onSearchChange?.(e.target.value)}
                        size="large"
                        allowClear
                        enterButton={false}
                        style={{
                            flex: 1,
                            minWidth: '200px',
                            maxWidth: '600px'
                        }}
                    />
                )}

                {/* Action Buttons */}
                {actions && (
                    <Space size="middle" style={{ minWidth: 'fit-content' }}>
                        {actions}
                    </Space>
                )}
            </div>

            {/* Mobile Layout - Stack search and buttons below title on small screens */}
            <style>{`
                @media (max-width: 768px) {
                    .ant-typography {
                        width: 100%;
                    }
                    .ant-input-search {
                        width: 100% !important;
                        max-width: 100% !important;
                        margin-top: 12px;
                    }
                    .ant-space {
                        width: 100%;
                        justify-content: flex-start;
                        margin-top: 12px;
                    }
                }
            `}</style>
        </div>
    );
};
