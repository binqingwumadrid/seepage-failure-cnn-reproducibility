function summary = generate_pseudocolor_images(base_dir, field_name, varargin)
%GENERATE_PSEUDOCOLOR_IMAGES Convert SPH particle results to RGB pseudocolor images.
%
% This implementation is compatible with MATLAB R2016a. It reads the
% physical parameters from each CSV file instead of inferring them from the
% run number. A manifest is written beside the generated images so that the
% exact source file, parameters, normalization range, and noise seed remain
% traceable.
%
% Example (pressure images, no added noise):
%   generate_pseudocolor_images(raw_dir, 'pressure');
%
% Example (equivalent plastic strain at several noise levels):
%   generate_pseudocolor_images(raw_dir, 'equivalent_plastic_strain', ...
%       'NoisePercent', [0 1 3 5]);
%
% Required inputs
%   base_dir   Directory containing dabashenliu_run_001, ..., run_1000.
%   field_name 'pressure' or 'equivalent_plastic_strain'.
%
% Name-value options
%   'RunIds'             Run identifiers to process (default: 1:1000).
%   'InputFile'          CSV snapshot name (default:
%                        particle_positions_150000.csv).
%   'OutputDir'          Output root (default: generated_<field> below
%                        base_dir).
%   'ParticleIds'        Particle material IDs (default: 1).
%   'NoisePercent'       Gaussian-noise standard deviations in percent of
%                        the normalized [0,1] range (default: 0).
%   'RandomSeed'         Base random seed (default: 2026).
%   'Nx', 'Ny'           Output width and height. Field-specific defaults
%                        are 800 by 300 for pressure and 1600 by 600 for
%                        equivalent plastic strain.
%   'XRange', 'YRange'   Physical plotting domain (default: [3 18], [0 6]).
%   'ValueRange'         Fixed field normalization range. Defaults are
%                        [0 30000] for pressure and [0 0.3] for strain.
%   'GaussianSigma'      Spatial smoothing sigma in pixels (default: 2).
%   'Overwrite'          Replace existing PNG files (default: false).

    if nargin < 2
        error('Both base_dir and field_name are required.');
    end

    if ~ischar(base_dir) || exist(base_dir, 'dir') ~= 7
        error('base_dir does not exist or is not a character path: %s', base_dir);
    end

    if ~ischar(field_name)
        error('field_name must be a character vector.');
    end

    field_name = lower(strtrim(field_name));
    if strcmp(field_name, 'pressure')
        default_value_range = [0 30000];
        default_nx = 800;
        default_ny = 300;
    elseif strcmp(field_name, 'equivalent_plastic_strain')
        default_value_range = [0 0.3];
        default_nx = 1600;
        default_ny = 600;
    else
        error(['Unsupported field_name: %s. Use pressure or ' ...
               'equivalent_plastic_strain.'], field_name);
    end

    parser = inputParser;
    parser.FunctionName = mfilename;
    addParameter(parser, 'RunIds', 1:1000, @(x) isnumeric(x) && isvector(x));
    addParameter(parser, 'InputFile', 'particle_positions_150000.csv', @ischar);
    addParameter(parser, 'OutputDir', '', @ischar);
    addParameter(parser, 'ParticleIds', 1, @(x) isnumeric(x) && isvector(x));
    addParameter(parser, 'NoisePercent', 0, @(x) isnumeric(x) && isvector(x) && all(x >= 0));
    addParameter(parser, 'RandomSeed', 2026, @(x) isnumeric(x) && isscalar(x));
    addParameter(parser, 'Nx', default_nx, @(x) isnumeric(x) && isscalar(x) && x >= 2);
    addParameter(parser, 'Ny', default_ny, @(x) isnumeric(x) && isscalar(x) && x >= 2);
    addParameter(parser, 'XRange', [3 18], @(x) isnumeric(x) && numel(x) == 2 && x(2) > x(1));
    addParameter(parser, 'YRange', [0 6], @(x) isnumeric(x) && numel(x) == 2 && x(2) > x(1));
    addParameter(parser, 'ValueRange', default_value_range, @(x) isnumeric(x) && numel(x) == 2 && x(2) > x(1));
    addParameter(parser, 'GaussianSigma', 2, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(parser, 'Overwrite', false, @(x) islogical(x) && isscalar(x));
    parse(parser, varargin{:});
    opts = parser.Results;

    run_ids = opts.RunIds(:)';
    noise_levels = opts.NoisePercent(:)';
    if isempty(run_ids)
        error('RunIds cannot be empty.');
    end
    if isempty(noise_levels)
        error('NoisePercent cannot be empty.');
    end
    if any(run_ids < 1) || any(run_ids ~= floor(run_ids))
        error('RunIds must contain positive integers.');
    end

    if isempty(opts.OutputDir)
        output_root = fullfile(base_dir, ['generated_' field_name]);
    else
        output_root = opts.OutputDir;
    end
    if exist(output_root, 'dir') ~= 7
        mkdir(output_root);
    end

    for noise_index = 1:numel(noise_levels)
        output_dirs{noise_index} = fullfile( ... %#ok<AGROW>
            output_root, noise_folder_name(noise_levels(noise_index)));
        if exist(output_dirs{noise_index}, 'dir') ~= 7
            mkdir(output_dirs{noise_index});
        end
    end

    [x_grid, y_grid] = meshgrid( ...
        linspace(opts.XRange(1), opts.XRange(2), opts.Nx), ...
        linspace(opts.YRange(1), opts.YRange(2), opts.Ny));
    color_map = jet(256);

    max_rows = numel(run_ids) * numel(noise_levels);
    manifest_run_id = zeros(max_rows, 1);
    manifest_source = cell(max_rows, 1);
    manifest_output = cell(max_rows, 1);
    manifest_field = cell(max_rows, 1);
    manifest_cohesion = zeros(max_rows, 1);
    manifest_friction = zeros(max_rows, 1);
    manifest_ks = zeros(max_rows, 1);
    manifest_log10ks = zeros(max_rows, 1);
    manifest_noise = zeros(max_rows, 1);
    manifest_seed = zeros(max_rows, 1);
    manifest_particle_count = zeros(max_rows, 1);
    manifest_clipped_low = zeros(max_rows, 1);
    manifest_clipped_high = zeros(max_rows, 1);
    manifest_status = cell(max_rows, 1);
    manifest_row = 0;

    generated_count = 0;
    skipped_count = 0;

    for run_index = 1:numel(run_ids)
        run_id = run_ids(run_index);
        run_folder = sprintf('dabashenliu_run_%03d', run_id);
        csv_path = fullfile(base_dir, run_folder, opts.InputFile);
        if exist(csv_path, 'file') ~= 2
            error('Required source file is missing: %s', csv_path);
        end

        data = readtable(csv_path);
        required_columns = {'x', 'y', 'id', field_name, ...
            'frictional_angle', 'cohesion_0', 'ks'};
        assert_required_columns(data.Properties.VariableNames, ...
            required_columns, csv_path);

        particle_mask = ismember(double(data.id), double(opts.ParticleIds));
        x_pos = double(data.x(particle_mask));
        y_pos = double(data.y(particle_mask));
        field_values = double(data.(field_name)(particle_mask));
        cohesion_values = double(data.cohesion_0(particle_mask));
        friction_values = double(data.frictional_angle(particle_mask));
        ks_values = double(data.ks(particle_mask));

        finite_mask = isfinite(x_pos) & isfinite(y_pos) & isfinite(field_values);
        x_pos = x_pos(finite_mask);
        y_pos = y_pos(finite_mask);
        field_values = field_values(finite_mask);
        cohesion_values = cohesion_values(finite_mask);
        friction_values = friction_values(finite_mask);
        ks_values = ks_values(finite_mask);

        if numel(field_values) < 3
            error('Fewer than three valid particles were found in %s.', csv_path);
        end

        cohesion = require_constant(cohesion_values, 'cohesion_0', csv_path);
        friction = require_constant(friction_values, 'frictional_angle', csv_path);
        ks_value = require_constant(ks_values, 'ks', csv_path);
        if ks_value <= 0
            error('ks must be positive in %s.', csv_path);
        end
        log10ks = log10(ks_value);

        interpolant = scatteredInterpolant( ...
            x_pos, y_pos, field_values, 'natural', 'none');
        raw_grid = interpolant(x_grid, y_grid);
        valid_grid = isfinite(raw_grid);

        if opts.GaussianSigma > 0
            % Normalized convolution prevents NaN values outside the
            % particle domain from contaminating neighboring valid pixels.
            values_for_filter = raw_grid;
            values_for_filter(~valid_grid) = 0;
            filter_weights = double(valid_grid);
            filtered_values = imgaussfilt(values_for_filter, opts.GaussianSigma);
            filtered_weights = imgaussfilt(filter_weights, opts.GaussianSigma);
            smoothed_grid = filtered_values ./ max(filtered_weights, eps);
            smoothed_grid(~valid_grid) = NaN;
        else
            smoothed_grid = raw_grid;
        end

        normalized_grid = (smoothed_grid - opts.ValueRange(1)) ./ ...
            (opts.ValueRange(2) - opts.ValueRange(1));
        clipped_low = sum(normalized_grid(valid_grid) < 0);
        clipped_high = sum(normalized_grid(valid_grid) > 1);
        normalized_grid(valid_grid & normalized_grid < 0) = 0;
        normalized_grid(valid_grid & normalized_grid > 1) = 1;

        output_filename = sprintf( ...
            'c%.0f_phi%.2f_logk%.6f.png', ...
            cohesion, friction, log10ks);

        for noise_index = 1:numel(noise_levels)
            noise_percent = noise_levels(noise_index);
            noise_seed = opts.RandomSeed + run_id * 10000 + ...
                round(noise_percent * 100);
            noisy_grid = normalized_grid;
            if noise_percent > 0
                rng(noise_seed, 'twister');
                noise = (noise_percent / 100) .* randn(size(noisy_grid));
                noisy_grid(valid_grid) = noisy_grid(valid_grid) + noise(valid_grid);
                noisy_grid(valid_grid & noisy_grid < 0) = 0;
                noisy_grid(valid_grid & noisy_grid > 1) = 1;
            end

            output_path = fullfile(output_dirs{noise_index}, output_filename);
            if exist(output_path, 'file') == 2 && ~opts.Overwrite
                status = 'skipped_existing';
                skipped_count = skipped_count + 1;
            else
                rgb_image = pseudocolor_rgb(noisy_grid, valid_grid, color_map);
                imwrite(rgb_image, output_path);
                status = 'generated';
                generated_count = generated_count + 1;
            end

            manifest_row = manifest_row + 1;
            manifest_run_id(manifest_row) = run_id;
            manifest_source{manifest_row} = fullfile(run_folder, opts.InputFile);
            % Store a path relative to output_root. This keeps the manifest
            % portable and avoids encoding host-specific directory names.
            manifest_output{manifest_row} = fullfile( ...
                noise_folder_name(noise_percent), output_filename);
            manifest_field{manifest_row} = field_name;
            manifest_cohesion(manifest_row) = cohesion;
            manifest_friction(manifest_row) = friction;
            manifest_ks(manifest_row) = ks_value;
            manifest_log10ks(manifest_row) = log10ks;
            manifest_noise(manifest_row) = noise_percent;
            manifest_seed(manifest_row) = noise_seed;
            manifest_particle_count(manifest_row) = numel(field_values);
            manifest_clipped_low(manifest_row) = clipped_low;
            manifest_clipped_high(manifest_row) = clipped_high;
            manifest_status{manifest_row} = status;
        end

        fprintf('Processed run %d/%d: %s\n', ...
            run_index, numel(run_ids), output_filename);
    end

    used = 1:manifest_row;
    manifest = table( ...
        manifest_run_id(used), manifest_source(used), manifest_output(used), ...
        manifest_field(used), manifest_cohesion(used), ...
        manifest_friction(used), manifest_ks(used), ...
        manifest_log10ks(used), manifest_noise(used), ...
        manifest_seed(used), manifest_particle_count(used), ...
        manifest_clipped_low(used), manifest_clipped_high(used), ...
        manifest_status(used), ...
        'VariableNames', {'run_id', 'source_csv', 'output_png', 'field', ...
        'cohesion_0', 'frictional_angle', 'ks', 'log10_ks', ...
        'noise_percent', 'noise_seed', 'particle_count', ...
        'pixels_clipped_low', 'pixels_clipped_high', 'status'});

    manifest_path = fullfile(output_root, 'generation_manifest.csv');
    writetable(manifest, manifest_path);

    summary = struct();
    summary.output_root = output_root;
    summary.manifest_path = manifest_path;
    summary.requested_runs = numel(run_ids);
    summary.noise_levels = noise_levels;
    summary.generated_images = generated_count;
    summary.skipped_images = skipped_count;
    summary.total_manifest_rows = manifest_row;

    fprintf('\nImage generation complete.\n');
    fprintf('Generated images: %d\n', generated_count);
    fprintf('Skipped existing images: %d\n', skipped_count);
    fprintf('Manifest: %s\n', manifest_path);
end


function assert_required_columns(actual_columns, required_columns, csv_path)
    for column_index = 1:numel(required_columns)
        if ~any(strcmp(actual_columns, required_columns{column_index}))
            error('Required column "%s" is missing from %s.', ...
                required_columns{column_index}, csv_path);
        end
    end
end


function value = require_constant(values, parameter_name, csv_path)
    values = values(isfinite(values));
    if isempty(values)
        error('No finite %s values were found in %s.', parameter_name, csv_path);
    end
    value = values(1);
    tolerance = 1e-7 * max(1, abs(value));
    if any(abs(values - value) > tolerance)
        error('%s is not constant for the selected particles in %s.', ...
            parameter_name, csv_path);
    end
end


function folder_name = noise_folder_name(noise_percent)
    label = sprintf('%g', noise_percent);
    label(label == '.') = 'p';
    folder_name = ['noise_' label 'pct'];
end


function rgb_image = pseudocolor_rgb(normalized_grid, valid_grid, color_map)
    [height, width] = size(normalized_grid);
    rgb_double = ones(height, width, 3);
    color_index = floor(normalized_grid(valid_grid) .* 255) + 1;
    color_index = max(1, min(256, color_index));

    for channel = 1:3
        channel_data = ones(height, width);
        channel_values = color_map(color_index, channel);
        channel_data(valid_grid) = channel_values;
        rgb_double(:, :, channel) = channel_data;
    end

    rgb_double = flipud(rgb_double);
    rgb_image = uint8(round(255 .* rgb_double));
end
