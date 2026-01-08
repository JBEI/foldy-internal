import { FileInfo } from '../types/types';

const hasMsaDir = (key: string) => key.includes('/msa/') || key.startsWith('msa/');

const isBoltzMsaCsv = (key: string) =>
    key.includes('boltz/boltz_results_input/msa/') && key.toLowerCase().endsWith('.csv');

export const findPreexistingMsaPath = (files?: FileInfo[]): string | null => {
    if (!files || files.length === 0) {
        return null;
    }

    const keys = files.map((file) => file.key);
    const boltzInput = keys.find((key) =>
        key.includes('boltz/boltz_results_input/msa/input_0.csv')
    );
    if (boltzInput) {
        return boltzInput;
    }

    const boltzCsv = keys.find(isBoltzMsaCsv);
    if (boltzCsv) {
        return boltzCsv;
    }

    const a3m = keys.find((key) => key.toLowerCase().endsWith('.a3m'));
    if (a3m) {
        return a3m;
    }

    const a2m = keys.find((key) => key.toLowerCase().endsWith('.a2m'));
    if (a2m) {
        return a2m;
    }

    const msaCsv = keys.find((key) => hasMsaDir(key) && key.toLowerCase().endsWith('.csv'));
    if (msaCsv) {
        return msaCsv;
    }

    return null;
};
