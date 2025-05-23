import React, { ReactNode, CSSProperties } from 'react';

interface PlotContainerProps {
    title?: string;
    children: ReactNode;
    style?: CSSProperties;
    height?: string;
    backgroundColor?: string;
}

export const PlotContainer: React.FC<PlotContainerProps> = ({
    title,
    children,
    style = {},
    height = '450px',
    backgroundColor = '#f9f9f9'
}) => {
    const containerStyle: CSSProperties = {
        marginTop: '30px',
        backgroundColor: '#fff',
        borderRadius: '8px',
        border: '1px solid #e0e0e0',
        padding: '20px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        ...style
    };

    const titleStyle: CSSProperties = {
        marginBottom: '20px',
        color: '#333',
        borderBottom: '1px solid #eee',
        paddingBottom: '10px'
    };

    const plotStyle: CSSProperties = {
        height,
        backgroundColor,
        padding: '15px',
        borderRadius: '4px'
    };

    return (
        <div style={containerStyle}>
            {title && <h3 style={titleStyle}>{title}</h3>}
            <div style={plotStyle}>
                {children}
            </div>
        </div>
    );
};

interface MetricsPlotContainerProps {
    children: ReactNode;
    style?: CSSProperties;
}

export const MetricsPlotContainer: React.FC<MetricsPlotContainerProps> = ({
    children,
    style = {}
}) => {
    return (
        <PlotContainer 
            title="Training Metrics"
            style={style}
        >
            {children}
        </PlotContainer>
    );
};

interface DataTableContainerProps {
    children: ReactNode;
    style?: CSSProperties;
}

export const DataTableContainer: React.FC<DataTableContainerProps> = ({
    children,
    style = {}
}) => {
    const defaultStyle: CSSProperties = {
        width: "auto",
        height: "auto",
        marginTop: "20px",
        ...style
    };

    return <div style={defaultStyle}>{children}</div>;
};