"""Optimized documentation generation system - zero redundancy design"""

import time
import os
import inspect
from typing import Dict, List, Any, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import json
import ast


def _current_time() -> float:
    """Shared time function"""
    return time.time()


class DocumentationType(Enum):
    """Optimized documentation type enum"""
    API = "api"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    CONFIGURATION = "configuration"
    TUTORIAL = "tutorial"


class OutputFormat(Enum):
    """Optimized output format enum"""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    RST = "rst"


@dataclass
class DocumentationSection:
    """Optimized documentation section"""
    title: str
    content: str
    section_type: DocumentationType
    level: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    subsections: List['DocumentationSection'] = field(default_factory=list)


@dataclass
class FunctionDocumentation:
    """Optimized function documentation"""
    name: str
    signature: str
    docstring: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    returns: Optional[Dict[str, Any]] = None
    raises: List[Dict[str, Any]] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    source_file: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class ClassDocumentation:
    """Optimized class documentation"""
    name: str
    docstring: str
    base_classes: List[str] = field(default_factory=list)
    methods: List[FunctionDocumentation] = field(default_factory=list)
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    properties: List[Dict[str, Any]] = field(default_factory=list)
    source_file: Optional[str] = None
    line_number: Optional[int] = None


class DocumentationGenerator:
    """Optimized documentation generator with zero redundancy"""

    def __init__(self, project_root: str, output_dir: str = "docs"):
        """
        Initialize documentation generator

        Args:
            project_root: Root directory of the project
            output_dir: Output directory for generated documentation
        """
        self.project_root = project_root
        self.output_dir = output_dir

        # Optimized: Documentation cache
        self._doc_cache: Dict[str, DocumentationSection] = {}
        self._function_cache: Dict[str, FunctionDocumentation] = {}
        self._class_cache: Dict[str, ClassDocumentation] = {}

        # Optimized: Template system
        self._templates = {
            OutputFormat.MARKDOWN: self._get_markdown_templates(),
            OutputFormat.HTML: self._get_html_templates(),
            OutputFormat.RST: self._get_rst_templates()
        }

        # Optimized: Configuration
        self._config = {
            'include_private': False,
            'include_source_links': True,
            'auto_generate_toc': True,
            'include_inheritance_diagrams': False,
            'max_line_length': 80
        }

    def generate_module_documentation(self, module_path: str,
                                    include_submodules: bool = True) -> DocumentationSection:
        """
        Generate documentation for a module - optimized module documentation

        Args:
            module_path: Path to the module
            include_submodules: Whether to include submodules

        Returns:
            Documentation section for the module
        """
        cache_key = f"module_{module_path}_{include_submodules}"
        if cache_key in self._doc_cache:
            return self._doc_cache[cache_key]

        try:
            # Import module dynamically
            module = self._import_module(module_path)
            if not module:
                raise ImportError(f"Failed to import module: {module_path}")

            # Extract module information
            module_doc = DocumentationSection(
                title=f"Module: {module_path}",
                content=self._extract_module_docstring(module),
                section_type=DocumentationType.MODULE,
                metadata={
                    'module_path': module_path,
                    'file_path': getattr(module, '__file__', ''),
                    'generation_time': _current_time()
                }
            )

            # Add functions documentation
            functions = self._extract_module_functions(module)
            if functions:
                functions_section = DocumentationSection(
                    title="Functions",
                    content="",
                    section_type=DocumentationType.FUNCTION,
                    level=2
                )

                for func_doc in functions:
                    func_section = self._create_function_section(func_doc)
                    functions_section.subsections.append(func_section)

                module_doc.subsections.append(functions_section)

            # Add classes documentation
            classes = self._extract_module_classes(module)
            if classes:
                classes_section = DocumentationSection(
                    title="Classes",
                    content="",
                    section_type=DocumentationType.CLASS,
                    level=2
                )

                for class_doc in classes:
                    class_section = self._create_class_section(class_doc)
                    classes_section.subsections.append(class_section)

                module_doc.subsections.append(classes_section)

            # Add submodules if requested
            if include_submodules:
                submodules = self._find_submodules(module_path)
                for submodule_path in submodules:
                    submodule_doc = self.generate_module_documentation(submodule_path, False)
                    module_doc.subsections.append(submodule_doc)

            # Cache and return
            self._doc_cache[cache_key] = module_doc
            return module_doc

        except Exception as e:
            # Return error documentation
            return DocumentationSection(
                title=f"Module: {module_path} (Error)",
                content=f"Failed to generate documentation: {e}",
                section_type=DocumentationType.MODULE,
                metadata={'error': str(e)}
            )

    def _import_module(self, module_path: str) -> Optional[Any]:
        """Import module dynamically - optimized import"""
        try:
            # Handle relative imports
            if module_path.startswith('.'):
                # Convert to absolute import
                package_parts = self.project_root.split(os.sep)
                if package_parts:
                    base_package = package_parts[-1]
                    module_path = f"{base_package}{module_path}"

            # Import using importlib
            import importlib
            return importlib.import_module(module_path)

        except ImportError as e:
            print(f"Failed to import {module_path}: {e}")
            return None

    def _extract_module_docstring(self, module: Any) -> str:
        """Extract module docstring - optimized extraction"""
        docstring = getattr(module, '__doc__', '')
        return self._clean_docstring(docstring) if docstring else ""

    def _extract_module_functions(self, module: Any) -> List[FunctionDocumentation]:
        """Extract module functions - optimized function extraction"""
        functions = []

        for name, obj in inspect.getmembers(module, inspect.isfunction):
            # Skip private functions if configured
            if not self._config['include_private'] and name.startswith('_'):
                continue

            # Skip imported functions
            if getattr(obj, '__module__', '') != module.__name__:
                continue

            func_doc = self._extract_function_documentation(obj, name)
            if func_doc:
                functions.append(func_doc)

        return functions

    def _extract_module_classes(self, module: Any) -> List[ClassDocumentation]:
        """Extract module classes - optimized class extraction"""
        classes = []

        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip private classes if configured
            if not self._config['include_private'] and name.startswith('_'):
                continue

            # Skip imported classes
            if getattr(obj, '__module__', '') != module.__name__:
                continue

            class_doc = self._extract_class_documentation(obj, name)
            if class_doc:
                classes.append(class_doc)

        return classes

    def _extract_function_documentation(self, func: Callable, name: str) -> Optional[FunctionDocumentation]:
        """Extract function documentation - optimized function analysis"""
        cache_key = f"func_{id(func)}"
        if cache_key in self._function_cache:
            return self._function_cache[cache_key]

        try:
            # Get function signature
            signature = str(inspect.signature(func))

            # Get docstring
            docstring = self._clean_docstring(inspect.getdoc(func) or "")

            # Parse docstring for parameters, returns, raises
            parsed_doc = self._parse_docstring(docstring)

            # Get source information
            source_file = None
            line_number = None
            try:
                source_file = inspect.getfile(func)
                line_number = inspect.getsourcelines(func)[1]
            except (OSError, TypeError):
                pass

            func_doc = FunctionDocumentation(
                name=name,
                signature=signature,
                docstring=docstring,
                parameters=parsed_doc.get('parameters', []),
                returns=parsed_doc.get('returns'),
                raises=parsed_doc.get('raises', []),
                examples=parsed_doc.get('examples', []),
                source_file=source_file,
                line_number=line_number
            )

            self._function_cache[cache_key] = func_doc
            return func_doc

        except Exception as e:
            print(f"Error extracting documentation for function {name}: {e}")
            return None

    def _extract_class_documentation(self, cls: Type, name: str) -> Optional[ClassDocumentation]:
        """Extract class documentation - optimized class analysis"""
        cache_key = f"class_{id(cls)}"
        if cache_key in self._class_cache:
            return self._class_cache[cache_key]

        try:
            # Get class docstring
            docstring = self._clean_docstring(inspect.getdoc(cls) or "")

            # Get base classes
            base_classes = [base.__name__ for base in cls.__bases__ if base != object]

            # Get methods
            methods = []
            for method_name, method_obj in inspect.getmembers(cls, inspect.ismethod):
                if not self._config['include_private'] and method_name.startswith('_'):
                    continue

                method_doc = self._extract_function_documentation(method_obj, method_name)
                if method_doc:
                    methods.append(method_doc)

            # Get regular functions (staticmethod, classmethod)
            for func_name, func_obj in inspect.getmembers(cls, inspect.isfunction):
                if not self._config['include_private'] and func_name.startswith('_'):
                    continue

                func_doc = self._extract_function_documentation(func_obj, func_name)
                if func_doc:
                    methods.append(func_doc)

            # Get attributes
            attributes = self._extract_class_attributes(cls)

            # Get properties
            properties = self._extract_class_properties(cls)

            # Get source information
            source_file = None
            line_number = None
            try:
                source_file = inspect.getfile(cls)
                line_number = inspect.getsourcelines(cls)[1]
            except (OSError, TypeError):
                pass

            class_doc = ClassDocumentation(
                name=name,
                docstring=docstring,
                base_classes=base_classes,
                methods=methods,
                attributes=attributes,
                properties=properties,
                source_file=source_file,
                line_number=line_number
            )

            self._class_cache[cache_key] = class_doc
            return class_doc

        except Exception as e:
            print(f"Error extracting documentation for class {name}: {e}")
            return None

    def _extract_class_attributes(self, cls: Type) -> List[Dict[str, Any]]:
        """Extract class attributes - optimized attribute extraction"""
        attributes = []

        # Get class annotations
        annotations = getattr(cls, '__annotations__', {})

        for attr_name, attr_type in annotations.items():
            if not self._config['include_private'] and attr_name.startswith('_'):
                continue

            attribute = {
                'name': attr_name,
                'type': self._format_type(attr_type),
                'default': getattr(cls, attr_name, None),
                'description': ""  # Could be extracted from docstring
            }
            attributes.append(attribute)

        return attributes

    def _extract_class_properties(self, cls: Type) -> List[Dict[str, Any]]:
        """Extract class properties - optimized property extraction"""
        properties = []

        for prop_name, prop_obj in inspect.getmembers(cls, lambda x: isinstance(x, property)):
            if not self._config['include_private'] and prop_name.startswith('_'):
                continue

            prop_doc = self._clean_docstring(inspect.getdoc(prop_obj) or "")

            property_info = {
                'name': prop_name,
                'type': 'property',
                'readable': prop_obj.fget is not None,
                'writable': prop_obj.fset is not None,
                'deletable': prop_obj.fdel is not None,
                'description': prop_doc
            }
            properties.append(property_info)

        return properties

    def _parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """Parse docstring for structured information - optimized parsing"""
        parsed = {
            'parameters': [],
            'returns': None,
            'raises': [],
            'examples': []
        }

        if not docstring:
            return parsed

        lines = docstring.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            line = line.strip()

            # Detect section headers
            if line.lower().startswith('args:') or line.lower().startswith('parameters:'):
                current_section = 'parameters'
                current_content = []
            elif line.lower().startswith('returns:') or line.lower().startswith('return:'):
                current_section = 'returns'
                current_content = []
            elif line.lower().startswith('raises:') or line.lower().startswith('exceptions:'):
                current_section = 'raises'
                current_content = []
            elif line.lower().startswith('example'):
                current_section = 'examples'
                current_content = []
            elif line and current_section:
                current_content.append(line)
            elif not line and current_content:
                # End of section
                self._process_docstring_section(current_section, current_content, parsed)
                current_section = None
                current_content = []

        # Process final section
        if current_section and current_content:
            self._process_docstring_section(current_section, current_content, parsed)

        return parsed

    def _process_docstring_section(self, section: str, content: List[str],
                                 parsed: Dict[str, Any]) -> None:
        """Process docstring section - optimized section processing"""
        if section == 'parameters':
            for line in content:
                if ':' in line:
                    parts = line.split(':', 1)
                    param_name = parts[0].strip()
                    param_desc = parts[1].strip() if len(parts) > 1 else ""

                    # Extract type information
                    param_type = ""
                    if '(' in param_name and ')' in param_name:
                        type_start = param_name.find('(')
                        type_end = param_name.find(')')
                        param_type = param_name[type_start+1:type_end]
                        param_name = param_name[:type_start].strip()

                    parsed['parameters'].append({
                        'name': param_name,
                        'type': param_type,
                        'description': param_desc
                    })

        elif section == 'returns':
            parsed['returns'] = {
                'type': "",
                'description': ' '.join(content)
            }

        elif section == 'raises':
            for line in content:
                if ':' in line:
                    parts = line.split(':', 1)
                    exception_type = parts[0].strip()
                    exception_desc = parts[1].strip() if len(parts) > 1 else ""

                    parsed['raises'].append({
                        'type': exception_type,
                        'description': exception_desc
                    })

        elif section == 'examples':
            parsed['examples'].append('\n'.join(content))

    def _clean_docstring(self, docstring: str) -> str:
        """Clean and format docstring - optimized cleaning"""
        if not docstring:
            return ""

        # Remove leading/trailing whitespace
        docstring = docstring.strip()

        # Handle indentation
        lines = docstring.split('\n')
        if len(lines) > 1:
            # Find minimum indentation (excluding first line)
            min_indent = float('inf')
            for line in lines[1:]:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)

            # Remove common indentation
            if min_indent < float('inf'):
                lines = [lines[0]] + [line[min_indent:] if line.strip() else line for line in lines[1:]]

        return '\n'.join(lines)

    def _format_type(self, type_hint: Any) -> str:
        """Format type hint as string - optimized type formatting"""
        if hasattr(type_hint, '__name__'):
            return type_hint.__name__
        else:
            return str(type_hint)

    def _find_submodules(self, module_path: str) -> List[str]:
        """Find submodules of a module - optimized submodule discovery"""
        submodules = []

        try:
            # Convert module path to file path
            module_parts = module_path.split('.')
            module_dir = os.path.join(self.project_root, *module_parts)

            if os.path.isdir(module_dir):
                for item in os.listdir(module_dir):
                    item_path = os.path.join(module_dir, item)

                    if os.path.isfile(item_path) and item.endswith('.py') and item != '__init__.py':
                        # Python file
                        submodule_name = item[:-3]  # Remove .py extension
                        submodules.append(f"{module_path}.{submodule_name}")

                    elif os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, '__init__.py')):
                        # Python package
                        submodules.append(f"{module_path}.{item}")

        except Exception as e:
            print(f"Error finding submodules for {module_path}: {e}")

        return submodules

    def _create_function_section(self, func_doc: FunctionDocumentation) -> DocumentationSection:
        """Create documentation section for function - optimized section creation"""
        content_parts = []

        # Function signature
        content_parts.append(f"```python\n{func_doc.name}{func_doc.signature}\n```")

        # Description
        if func_doc.docstring:
            content_parts.append(func_doc.docstring)

        # Parameters
        if func_doc.parameters:
            content_parts.append("**Parameters:**")
            for param in func_doc.parameters:
                param_line = f"- `{param['name']}`"
                if param['type']:
                    param_line += f" ({param['type']})"
                if param['description']:
                    param_line += f": {param['description']}"
                content_parts.append(param_line)

        # Returns
        if func_doc.returns:
            content_parts.append("**Returns:**")
            return_line = ""
            if func_doc.returns['type']:
                return_line += f"({func_doc.returns['type']}) "
            return_line += func_doc.returns['description']
            content_parts.append(return_line)

        # Raises
        if func_doc.raises:
            content_parts.append("**Raises:**")
            for exc in func_doc.raises:
                exc_line = f"- `{exc['type']}`"
                if exc['description']:
                    exc_line += f": {exc['description']}"
                content_parts.append(exc_line)

        # Examples
        if func_doc.examples:
            content_parts.append("**Examples:**")
            for example in func_doc.examples:
                content_parts.append(f"```python\n{example}\n```")

        return DocumentationSection(
            title=func_doc.name,
            content='\n\n'.join(content_parts),
            section_type=DocumentationType.FUNCTION,
            level=3,
            metadata={
                'source_file': func_doc.source_file,
                'line_number': func_doc.line_number
            }
        )

    def _create_class_section(self, class_doc: ClassDocumentation) -> DocumentationSection:
        """Create documentation section for class - optimized section creation"""
        content_parts = []

        # Class header
        class_header = f"class {class_doc.name}"
        if class_doc.base_classes:
            class_header += f"({', '.join(class_doc.base_classes)})"

        content_parts.append(f"```python\n{class_header}\n```")

        # Description
        if class_doc.docstring:
            content_parts.append(class_doc.docstring)

        # Create class section
        class_section = DocumentationSection(
            title=class_doc.name,
            content='\n\n'.join(content_parts),
            section_type=DocumentationType.CLASS,
            level=3,
            metadata={
                'source_file': class_doc.source_file,
                'line_number': class_doc.line_number
            }
        )

        # Add attributes section
        if class_doc.attributes:
            attr_content = []
            for attr in class_doc.attributes:
                attr_line = f"- `{attr['name']}`"
                if attr['type']:
                    attr_line += f" ({attr['type']})"
                if attr['description']:
                    attr_line += f": {attr['description']}"
                attr_content.append(attr_line)

            attr_section = DocumentationSection(
                title="Attributes",
                content='\n'.join(attr_content),
                section_type=DocumentationType.CLASS,
                level=4
            )
            class_section.subsections.append(attr_section)

        # Add properties section
        if class_doc.properties:
            prop_content = []
            for prop in class_doc.properties:
                prop_line = f"- `{prop['name']}` (property)"
                if prop['description']:
                    prop_line += f": {prop['description']}"
                prop_content.append(prop_line)

            prop_section = DocumentationSection(
                title="Properties",
                content='\n'.join(prop_content),
                section_type=DocumentationType.CLASS,
                level=4
            )
            class_section.subsections.append(prop_section)

        # Add methods section
        if class_doc.methods:
            methods_section = DocumentationSection(
                title="Methods",
                content="",
                section_type=DocumentationType.FUNCTION,
                level=4
            )

            for method_doc in class_doc.methods:
                method_section = self._create_function_section(method_doc)
                method_section.level = 5
                methods_section.subsections.append(method_section)

            class_section.subsections.append(methods_section)

        return class_section

    def _get_markdown_templates(self) -> Dict[str, str]:
        """Get Markdown templates - optimized template system"""
        return {
            'header': "# {title}\n\n{content}\n\n",
            'section': "{'#' * level} {title}\n\n{content}\n\n",
            'toc': "## Table of Contents\n\n{toc_items}\n\n",
            'toc_item': "- [{title}](#{anchor})\n"
        }

    def _get_html_templates(self) -> Dict[str, str]:
        """Get HTML templates - optimized template system"""
        return {
            'header': "<h1>{title}</h1>\n<div>{content}</div>\n",
            'section': "<h{level}>{title}</h{level}>\n<div>{content}</div>\n",
            'toc': "<h2>Table of Contents</h2>\n<ul>{toc_items}</ul>\n",
            'toc_item': '<li><a href="#{anchor}">{title}</a></li>\n'
        }

    def _get_rst_templates(self) -> Dict[str, str]:
        """Get RST templates - optimized template system"""
        return {
            'header': "{title}\n{'=' * len(title)}\n\n{content}\n\n",
            'section': "{title}\n{'-' * len(title)}\n\n{content}\n\n",
            'toc': ".. contents:: Table of Contents\n   :depth: 2\n\n"
        }

    def render_documentation(self, doc_section: DocumentationSection,
                           output_format: OutputFormat = OutputFormat.MARKDOWN) -> str:
        """
        Render documentation section to specified format - optimized rendering

        Args:
            doc_section: Documentation section to render
            output_format: Output format

        Returns:
            Rendered documentation string
        """
        templates = self._templates.get(output_format, self._templates[OutputFormat.MARKDOWN])

        # Render main section
        if doc_section.level == 1:
            template = templates['header']
        else:
            template = templates['section']

        # Render content
        rendered_content = template.format(
            title=doc_section.title,
            content=doc_section.content,
            level=doc_section.level
        )

        # Render subsections
        for subsection in doc_section.subsections:
            rendered_content += self.render_documentation(subsection, output_format)

        return rendered_content

    def generate_table_of_contents(self, doc_section: DocumentationSection,
                                 output_format: OutputFormat = OutputFormat.MARKDOWN) -> str:
        """Generate table of contents - optimized TOC generation"""
        templates = self._templates.get(output_format, self._templates[OutputFormat.MARKDOWN])

        toc_items = []
        self._collect_toc_items(doc_section, toc_items, output_format)

        if 'toc' in templates:
            if output_format == OutputFormat.MARKDOWN:
                toc_content = ''.join(toc_items)
            else:
                toc_content = ''.join(toc_items)

            return templates['toc'].format(toc_items=toc_content)

        return ""

    def _collect_toc_items(self, section: DocumentationSection, toc_items: List[str],
                          output_format: OutputFormat, level: int = 0) -> None:
        """Collect table of contents items - optimized TOC collection"""
        templates = self._templates.get(output_format, self._templates[OutputFormat.MARKDOWN])

        if level > 0:  # Skip root level
            anchor = section.title.lower().replace(' ', '-').replace(':', '')
            indent = '  ' * (level - 1)

            if output_format == OutputFormat.MARKDOWN:
                toc_items.append(f"{indent}- [{section.title}](#{anchor})\n")
            elif 'toc_item' in templates:
                toc_items.append(templates['toc_item'].format(
                    title=section.title,
                    anchor=anchor
                ))

        for subsection in section.subsections:
            self._collect_toc_items(subsection, toc_items, output_format, level + 1)

    def save_documentation(self, doc_section: DocumentationSection,
                          file_path: str, output_format: OutputFormat = OutputFormat.MARKDOWN) -> bool:
        """
        Save documentation to file - optimized file saving

        Args:
            doc_section: Documentation section to save
            file_path: Output file path
            output_format: Output format

        Returns:
            True if successful
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(file_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Render documentation
            rendered_content = ""

            # Add table of contents if configured
            if self._config['auto_generate_toc']:
                toc = self.generate_table_of_contents(doc_section, output_format)
                if toc:
                    rendered_content += toc + "\n"

            # Add main content
            rendered_content += self.render_documentation(doc_section, output_format)

            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(rendered_content)

            return True

        except Exception as e:
            print(f"Failed to save documentation to {file_path}: {e}")
            return False

    def clear_cache(self) -> None:
        """Clear documentation cache"""
        self._doc_cache.clear()
        self._function_cache.clear()
        self._class_cache.clear()