// frontend/src/components/UserSelector.tsx
import React, { useState, useCallback, forwardRef } from 'react';
import AsyncSelect from 'react-select/async'; // Import AsyncSelect
import { SingleValue } from 'react-select';
import { searchUsers } from '../services/recommendationService'; // Import the NEW search function
import { User } from '../types';
import { debounce } from 'lodash'; // Import debounce

// Define the structure for react-select options
interface UserOption {
  value: number;
  label: string;
}

interface UserSelectorProps {
  onUserSelect: (userId: number | null) => void; // Allow null for clearing selection
  isLoading: boolean; // To disable selector during parent (recommendation fetch) loading
}

const UserSelector = forwardRef<any, UserSelectorProps>(({ onUserSelect, isLoading }, ref) => {
  // State to hold the currently selected user option (controlled component)
  const [selectedOption, setSelectedOption] = useState<SingleValue<UserOption>>(null);

  // --- Debounced Function to Fetch Users ---
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const debouncedLoadUsers = useCallback(
    debounce(
      async (
        inputValue: string,
        callback: (options: UserOption[]) => void
      ) => {
        if (!inputValue || inputValue.length < 1) { // Require at least 1 character
          callback([]); // Return empty options if input is too short
          return;
        }
        try {
          console.log(`Searching users with query: "${inputValue}"`);
          const users: User[] = await searchUsers(inputValue); // Call the API service
          const options = users.map(user => ({
            value: user.student_id,
            label: String(user.student_id) // Label must be a string
          }));
          callback(options); // Pass formatted options to react-select
        } catch (error) {
          console.error("Failed to load users via search:", error);
          callback([]); // Return empty options on error
          // Optionally set an error state here to display a message
        }
      },
      350 // Debounce time in milliseconds (e.g., 350ms)
    ),
    [] // Empty dependency array for useCallback means the debounced function is created once
  );

  // --- loadOptions function for AsyncSelect ---
  const loadOptions = (
    inputValue: string,
    callback: (options: UserOption[]) => void
  ) => {
    // Call the debounced function
    debouncedLoadUsers(inputValue, callback);
  };

  // --- Handle selection change ---
  const handleChange = (selected: SingleValue<UserOption>) => {
    setSelectedOption(selected); // Update local state
    if (selected) {
      onUserSelect(selected.value); // Pass the numeric user ID up
    } else {
      onUserSelect(null); // Pass null up if selection is cleared
    }
  };

  // --- Custom Styles for React Select (Black Theme) ---
  const customSelectStyles = {
    control: (provided: any, state: any) => ({
      ...provided,
      backgroundColor: 'var(--color-surface)', // Use CSS variable or Tailwind color value
      borderColor: state.isFocused ? 'var(--color-primary)' : 'var(--color-border-color)',
      boxShadow: state.isFocused ? '0 0 0 1px var(--color-primary)' : 'none',
      color: 'var(--color-text-primary)',
      '&:hover': {
        borderColor: state.isFocused ? 'var(--color-primary)' : 'var(--color-border-color)',
      },
      borderRadius: '0.375rem', // Adjust as needed (tailwind rounded-md)
      minHeight: '42px', // Ensure consistent height
    }),
    menu: (provided: any) => ({
      ...provided,
      backgroundColor: 'var(--color-surface)',
      borderColor: 'var(--color-border-color)',
      borderWidth: '1px',
      borderRadius: '0.375rem',
      zIndex: 50, // Ensure dropdown is above other content
    }),
    option: (provided: any, state: any) => ({
      ...provided,
      backgroundColor: state.isSelected
        ? 'var(--color-primary)'
        : state.isFocused
        ? 'var(--color-border-color)' // Hover background
        : 'var(--color-surface)',
      color: state.isSelected ? '#fff' : 'var(--color-text-primary)', // Text color
      '&:active': {
        backgroundColor: 'var(--color-primary)', // Click background
      },
      cursor: 'pointer',
    }),
    singleValue: (provided: any) => ({
      ...provided,
      color: 'var(--color-text-primary)', // Color of the selected value text
    }),
    input: (provided: any) => ({
      ...provided,
      color: 'var(--color-text-primary)', // Color of the input text while typing
      margin: '0px',
      padding: '0px',
    }),
    placeholder: (provided: any) => ({
      ...provided,
      color: 'var(--color-text-muted)', // Color of the placeholder text
    }),
    indicatorSeparator: (provided: any) => ({
        ...provided,
        backgroundColor: 'var(--color-border-color)', // Color of the separator line
    }),
    dropdownIndicator: (provided: any, state: any) => ({
        ...provided,
        color: state.isFocused ? 'var(--color-primary)' : 'var(--color-text-muted)',
         '&:hover': {
            color: 'var(--color-primary)',
         },
         cursor: 'pointer',
    }),
    clearIndicator: (provided: any) => ({
        ...provided,
        color: 'var(--color-text-muted)',
        '&:hover': {
            color: 'var(--color-primary)', // Hover color for clear button
        },
        cursor: 'pointer',
    }),
    loadingIndicator: (provided: any) => ({
        ...provided,
        color: 'var(--color-primary)', // Color of the loading spinner
    }),
    noOptionsMessage: (provided: any) => ({
        ...provided,
        color: 'var(--color-text-muted)', // Color for 'No options' message
    }),
     valueContainer: (provided: any) => ({ // Ensure padding matches input height
        ...provided,
        padding: '2px 8px',
    }),
  };

  // --- Add CSS Variables (e.g., in index.css or a global style sheet) ---
  /* Add this to your src/index.css inside the @layer base block or globally */
  /*
  :root {
    --color-background: #111827;
    --color-surface: #1f2937;
    --color-primary: #06b6d4;
    --color-text-primary: #f9fafb;
    --color-text-secondary: #d1d5db;
    --color-text-muted: #9ca3af;
    --color-border-color: #374151;
  }
  */

  return (
    <div className="mb-8 p-6 bg-surface shadow-lg rounded-lg border border-border-color max-w-xl mx-auto"> {/* Center and style container */}
      <label htmlFor="userSelect" className="block text-sm font-medium text-text-secondary mb-2">
        Search and Select Student ID:
      </label>
      <AsyncSelect<UserOption> // Specify the type for AsyncSelect
        id="userSelect"
        cacheOptions // Cache results for the same search term
        loadOptions={loadOptions} // Use the debounced fetcher
        defaultOptions // Load default options on initial focus (optional, can be empty array or predefined list)
        value={selectedOption}
        onChange={handleChange}
        isDisabled={isLoading} // Disable if parent is loading recommendations
        placeholder={"Type to search student ID..."}
        isClearable={true}
        noOptionsMessage={({ inputValue }) =>
            !inputValue || inputValue.length < 1
             ? 'Please enter 1 or more characters'
             : 'No users found matching your query'
        }
        loadingMessage={() => 'Loading matching users...'}
        styles={customSelectStyles} // Apply custom styles
        instanceId="user-async-select" // Unique ID for accessibility
      />
      {/* Removed the old fetch error message display */}
    </div>
  );
});

export default UserSelector;