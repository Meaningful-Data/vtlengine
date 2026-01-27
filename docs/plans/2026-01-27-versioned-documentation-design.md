# Versioned Documentation Design

**Date**: 2026-01-27
**Issue**: #466
**Status**: Approved

## Overview

Implement versioned documentation for VTL Engine using sphinx-multiversion, providing users with a dropdown menu to access documentation for different released versions, with the latest stable release as the default landing page.

## Goals

1. Generate documentation for each git tag (version)
2. Provide version dropdown in documentation UI
3. Label pre-release versions (containing "rc")
4. Mark latest stable version
5. Default to latest stable release when accessing root URL
6. Maintain existing GitHub Pages deployment

## Architecture

### Version Selection Strategy

**Sources:**
- All git tags matching pattern `v*` (e.g., v1.5.0, v1.5.0rc6, v1.4.0)
- Main branch (development version)

**Version Labeling:**
- Tags with "rc" in name → labeled "(pre-release)"
- Highest stable version (no "rc") → labeled "(latest)"
- Main branch → labeled "(development)"

**URL Structure:**
```
https://meaningful-data.github.io/vtlengine/
  ├── index.html (redirects to latest stable)
  ├── v1.5.0/ (latest stable release)
  │   ├── index.html
  │   ├── api.html
  │   └── ...
  ├── v1.5.0rc6/ (pre-release)
  │   └── ...
  ├── v1.4.0/ (older stable)
  │   └── ...
  └── main/ (development)
      └── ...
```

### Technology Choice: sphinx-multiversion

**Why sphinx-multiversion:**
- Active maintenance and community support
- Native integration with sphinx_rtd_theme (already in use)
- Automatic version detection from git
- Built-in support for version labeling
- Works with existing GitHub Pages setup
- Single build command generates all versions

**Alternatives Considered:**
- sphinx-versions: Older, less maintained
- Read the Docs hosting: Requires infrastructure change

## Implementation Details

### 1. Sphinx Configuration Changes

**File**: `docs/conf.py`

**Changes:**
1. Add `sphinx_multiversion` to extensions
2. Configure version selection:
   ```python
   smv_tag_whitelist = r'^v\d+\.\d+\.\d+.*$'  # Matches v1.5.0, v1.5.0rc6, etc.
   smv_branch_whitelist = r'^(main)$'          # Only main branch
   smv_remote_whitelist = r'^.*$'              # Allow all remotes
   ```
3. Set latest version detection:
   ```python
   smv_latest_version = 'v1.5.0'  # Will be dynamically determined
   ```
4. Configure output structure:
   ```python
   smv_outputdir_format = '{ref.name}'  # Creates v1.5.0/, v1.4.0/, etc.
   ```

### 2. Template Customization

**File**: `docs/_templates/versioning.html`

Create custom template for version dropdown:
- Display in RTD theme sidebar
- Show all available versions sorted (latest first)
- Add labels: "(latest)", "(pre-release)", "(development)"
- Link to corresponding version directory

**File**: `docs/_templates/layout.html` (or extend existing)

Override RTD theme to include versioning template in sidebar.

### 3. Root Redirect Generation

**Purpose**: Redirect root URL to latest stable version

**Implementation**: Python script that:
1. Parses all version directories in `_site/`
2. Identifies latest stable version:
   - Filter out versions with "rc"
   - Filter out "main" branch
   - Sort by semantic version
   - Select highest
3. Generates `_site/index.html`:
   ```html
   <!DOCTYPE html>
   <html>
   <head>
       <meta http-equiv="refresh" content="0; url=./v1.5.0/">
       <title>Redirecting...</title>
   </head>
   <body>
       Redirecting to latest version...
   </body>
   </html>
   ```

**Edge Cases:**
- No stable versions exist (only pre-releases): Redirect to latest pre-release
- No versions at all: Show error message

### 4. GitHub Actions Workflow Updates

**File**: `.github/workflows/docs.yml`

**Changes:**

1. **Fetch full git history:**
   ```yaml
   - name: Checkout code
     uses: actions/checkout@v4
     with:
       fetch-depth: 0  # Fetch all history for all tags and branches
   ```

2. **Add sphinx-multiversion dependency:**
   - Update pyproject.toml or install directly in workflow
   ```yaml
   - name: Install dependencies
     run: |
       poetry install
       poetry add --group dev sphinx-multiversion
   ```

3. **Build with sphinx-multiversion:**
   ```yaml
   - name: Build with Sphinx Multi-Version
     run: |
       poetry run sphinx-multiversion docs _site
   ```

4. **Generate root redirect:**
   ```yaml
   - name: Generate root redirect
     run: |
       poetry run python scripts/generate_redirect.py
   ```

5. **Update validation:**
   ```yaml
   - name: Validate error messages documentation
     run: |
       # Check at least one version has error_messages.html
       if ! find _site/v* -name "error_messages.html" -type f | grep -q .; then
         echo "ERROR: No error_messages.html found in any version"
         exit 1
       fi
   ```

**Triggers:** (unchanged)
- On release publication
- Manual workflow dispatch

### 5. New Script: Redirect Generator

**File**: `scripts/generate_redirect.py`

**Purpose**: Create root index.html that redirects to latest stable version

**Algorithm:**
1. List directories in `_site/` matching `v*` pattern
2. Parse semantic versions, filtering out "rc" versions
3. Sort versions and select highest
4. Generate redirect HTML
5. Write to `_site/index.html`

**Dependencies**: Standard library only (pathlib, re, html)

## Dependencies

**New Dependencies:**
- `sphinx-multiversion` (development dependency)

**Existing Dependencies:**
- `sphinx` (already present)
- `sphinx_rtd_theme` (already present)
- Python 3.9+ (already required)

## Testing Strategy

### Manual Testing Checklist

1. **Build Verification:**
   - [ ] Run `sphinx-multiversion docs _site` locally
   - [ ] Verify all expected versions are built
   - [ ] Check version directories exist with correct names

2. **Version Dropdown:**
   - [ ] Verify dropdown appears in sidebar
   - [ ] Check all versions are listed
   - [ ] Verify labels: "(latest)", "(pre-release)", "(development)"
   - [ ] Test links navigate to correct versions

3. **Root Redirect:**
   - [ ] Access root URL redirects to latest stable
   - [ ] Verify correct version is selected
   - [ ] Test with only pre-releases present

4. **Content Verification:**
   - [ ] Check error_messages.html exists in each version
   - [ ] Verify API documentation is complete
   - [ ] Test all internal links within a version

5. **GitHub Actions:**
   - [ ] Trigger workflow manually
   - [ ] Verify build succeeds
   - [ ] Check deployment to GitHub Pages
   - [ ] Access deployed site and test navigation

### Edge Cases to Test

1. Only pre-release versions exist (no stable releases)
2. New tag is added (build should include it)
3. Main branch updates (development docs should update)
4. Theme customization persists across versions

## Migration Path

### Deployment Strategy

1. **Development:**
   - Implement changes in feature branch `cr-466`
   - Test locally with existing tags
   - Verify build output structure

2. **Testing:**
   - Deploy to test environment or GitHub Pages preview
   - Validate all versions accessible
   - Check dropdown functionality

3. **Production:**
   - Merge to main
   - Trigger workflow (manual or release)
   - Monitor deployment
   - Verify live site

### Rollback Plan

If issues arise after deployment:
1. Revert workflow to use `sphinx-build` (original command)
2. Re-deploy with single-version documentation
3. Investigate issues in separate branch
4. Re-deploy versioned docs when fixed

### Backwards Compatibility

**URL Changes:**
- Old: `https://meaningful-data.github.io/vtlengine/api.html`
- New: `https://meaningful-data.github.io/vtlengine/` → redirects → `https://meaningful-data.github.io/vtlengine/v1.5.0/api.html`

**Impact:**
- Root URL links remain valid (auto-redirect)
- Deep links to specific pages will break temporarily
- Solution: One-time redirection notice or update external links

**Recommendation**: Add a banner on the old deployment (before switching) announcing the URL structure change.

## Timeline Estimate

**Note**: Avoiding time estimates per project guidelines. Work broken into incremental steps for visibility.

## Implementation Checklist

**Phase 1: Configuration**
- [ ] Add sphinx-multiversion to dependencies
- [ ] Update docs/conf.py with multiversion configuration
- [ ] Create custom templates for version dropdown

**Phase 2: Build Script**
- [ ] Create scripts/generate_redirect.py
- [ ] Test redirect generation locally
- [ ] Add script to workflow

**Phase 3: Workflow Updates**
- [ ] Update GitHub Actions to fetch full history
- [ ] Change build command to use sphinx-multiversion
- [ ] Add redirect generation step
- [ ] Update validation steps

**Phase 4: Testing**
- [ ] Build locally with multiple versions
- [ ] Verify version dropdown works
- [ ] Test root redirect
- [ ] Validate all content accessible

**Phase 5: Deployment**
- [ ] Create pull request
- [ ] Run workflow in PR branch
- [ ] Deploy to GitHub Pages
- [ ] Verify production site

## Success Criteria

1. Documentation builds successfully for all git tags
2. Version dropdown visible in sidebar with correct labels
3. Root URL redirects to latest stable version
4. All versions independently accessible
5. Pre-releases labeled appropriately
6. GitHub Actions workflow completes without errors
7. No broken links within any version

## Future Enhancements

**Not in scope for #466:**
- Version comparison view
- Deprecation notices for old versions
- Automated link migration for old URLs
- Search across all versions
- Version-specific banners (e.g., "You're viewing old docs")

These can be addressed in future issues if needed.

## References

- [sphinx-multiversion documentation](https://holzhaus.github.io/sphinx-multiversion/)
- [Sphinx RTD Theme customization](https://sphinx-rtd-theme.readthedocs.io/)
- [GitHub Pages deployment](https://docs.github.com/en/pages)
- [VTL 2.1 Reference Manual](https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf)
