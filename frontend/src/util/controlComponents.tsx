import React, { CSSProperties } from 'react';

interface NumberInputControlProps {
    label: string;
    value: number;
    onChange: (value: number) => void;
    min?: number;
    max?: number;
    step?: number;
    style?: CSSProperties;
    inputWidth?: string;
}

export const NumberInputControl: React.FC<NumberInputControlProps> = ({
    label,
    value,
    onChange,
    min,
    max,
    step = 1,
    style = {},
    inputWidth = '100px'
}) => {
    const containerStyle: CSSProperties = {
        marginBottom: '10px',
        ...style
    };

    const inputStyle: CSSProperties = {
        width: inputWidth,
        marginLeft: '10px'
    };

    return (
        <div style={containerStyle}>
            <label>
                {label}:
                <input
                    type="number"
                    className="uk-input"
                    value={value}
                    onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
                    style={inputStyle}
                    min={min}
                    max={max}
                    step={step}
                />
            </label>
        </div>
    );
};

interface CheckboxControlProps {
    label: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
    style?: CSSProperties;
}

export const CheckboxControl: React.FC<CheckboxControlProps> = ({
    label,
    checked,
    onChange,
    style = {}
}) => {
    const containerStyle: CSSProperties = {
        marginBottom: '10px',
        ...style
    };

    return (
        <div style={containerStyle}>
            <label>
                <input
                    type="checkbox"
                    className="uk-checkbox"
                    checked={checked}
                    onChange={(e) => onChange(e.target.checked)}
                />
                {' '}{label}
            </label>
        </div>
    );
};

interface TextInputControlProps {
    label: string;
    value: string;
    onChange: (value: string) => void;
    placeholder?: string;
    style?: CSSProperties;
    inputStyle?: CSSProperties;
}

export const TextInputControl: React.FC<TextInputControlProps> = ({
    label,
    value,
    onChange,
    placeholder,
    style = {},
    inputStyle = {}
}) => {
    return (
        <div style={style}>
            <label className="uk-form-label">{label}</label>
            <input
                type="text"
                className="uk-input"
                value={value}
                onChange={(e) => onChange(e.target.value)}
                placeholder={placeholder}
                style={inputStyle}
            />
        </div>
    );
};

interface TextAreaControlProps {
    label: string;
    value: string;
    onChange: (value: string) => void;
    placeholder?: string;
    rows?: number;
    style?: CSSProperties;
    inputStyle?: CSSProperties;
}

export const TextAreaControl: React.FC<TextAreaControlProps> = ({
    label,
    value,
    onChange,
    placeholder,
    rows = 5,
    style = {},
    inputStyle = {}
}) => {
    return (
        <div style={style}>
            <label className="uk-form-label">{label}</label>
            <textarea
                className="uk-textarea"
                rows={rows}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                placeholder={placeholder}
                style={inputStyle}
            />
        </div>
    );
};

interface SelectControlProps {
    label: string;
    value: string;
    onChange: (value: string) => void;
    options: { value: string; label: string }[];
    style?: CSSProperties;
    selectStyle?: CSSProperties;
}

export const SelectControl: React.FC<SelectControlProps> = ({
    label,
    value,
    onChange,
    options,
    style = {},
    selectStyle = {}
}) => {
    return (
        <div style={style}>
            <label className="uk-form-label">{label}</label>
            <select
                className="uk-select"
                value={value}
                onChange={(e) => onChange(e.target.value)}
                style={selectStyle}
            >
                {options.map(option => (
                    <option key={option.value} value={option.value}>
                        {option.label}
                    </option>
                ))}
            </select>
        </div>
    );
};

interface FileUploadControlProps {
    label: string;
    onChange: (file: File | null) => void;
    accept?: string;
    selectedFile?: File | null;
    style?: CSSProperties;
}

export const FileUploadControl: React.FC<FileUploadControlProps> = ({
    label,
    onChange,
    accept,
    selectedFile,
    style = {}
}) => {
    return (
        <div style={style}>
            <label className="uk-form-label">{label}</label>
            <input
                type="file"
                accept={accept}
                onChange={(e) => onChange(e.target.files?.[0] || null)}
                className="uk-input"
            />
            {selectedFile && (
                <p className="uk-text-meta">Selected file: {selectedFile.name}</p>
            )}
        </div>
    );
};

interface MultiSelectControlProps {
    label: string;
    options: { key: string; label: string }[];
    selectedValues: string[];
    onChange: (selectedValues: string[]) => void;
    size?: number;
    style?: CSSProperties;
}

export const MultiSelectControl: React.FC<MultiSelectControlProps> = ({
    label,
    options,
    selectedValues,
    onChange,
    size,
    style = {}
}) => {
    const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const selectedOptions = Array.from(event.target.selectedOptions).map(option => option.value);
        onChange(selectedOptions);
    };

    return (
        <div style={style}>
            <label className="uk-form-label">{label}</label>
            <select
                className="uk-select"
                multiple
                size={size || Math.min(10, options.length || 1)}
                value={selectedValues}
                onChange={handleChange}
            >
                {options.map(option => (
                    <option key={option.key} value={option.key}>
                        {option.label}
                    </option>
                ))}
            </select>
            <p className="uk-text-meta">
                Selected {selectedValues.length} item(s)
            </p>
        </div>
    );
};